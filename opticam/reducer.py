from functools import partial
import json
import logging
from multiprocessing import cpu_count
import os
from typing import Callable, Dict, List, Literal, Tuple

from astropy.table import QTable
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
from photutils.segmentation import SourceCatalog, detect_threshold
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

from opticam.align import align_batch
from opticam.background.global_background import BaseBackground, DefaultBackground
from opticam.correctors.flat_field_corrector import FlatFieldCorrector
from opticam.finders import DefaultFinder, get_source_coords_from_image
from opticam.photometers import BasePhotometer, perform_photometry
from opticam.utils.batching import get_batches, get_batch_size
from opticam.utils.constants import bar_format, pixel_scales
from opticam.utils.data_checks import check_data
from opticam.plotting.gifs import compile_gif, create_gif_frame
from opticam.utils.helpers import camel_to_snake
from opticam.utils.fits_handlers import get_data, get_stacked_images, save_stacked_images
from opticam.utils.logging import recursive_log
from opticam.plotting.plots import plot_backgrounds, plot_background_meshes, plot_catalogs, plot_time_between_files


class Reducer:
    """
    Class for reducing OPTICAM data.
    """

    def __init__(
        self,
        out_directory: str,
        data_directory: None | str = None,
        c1_directory: None | str = None,
        c2_directory: None | str = None,
        c3_directory: None | str = None,
        rebin_factor: int = 1,
        flat_corrector: None | FlatFieldCorrector = None,
        background: BaseBackground | None = None,
        finder: None | Callable = None,
        threshold: float = 5,
        aperture_selector: Callable = np.median,
        remove_cosmic_rays: bool = False,
        number_of_processors: int = cpu_count() // 2,
        show_plots: bool = True,
        verbose: bool = True
        ) -> None:
        """
        Class for reducing OPTICAM data.
        
        Parameters
        ----------
        out_directory: str
            The path to the directory to save the output files.
        data_directory: str, optional
            The path to the directory containing the data, by default `None`. If `None`, any of c1_directory, c2_directory,
            or c3_directory must be defined. If data_directory is defined, c1_directory, c2_directory, and c3_directory
            are ignored.
        c1_directory: str, optional
            The path to the directory containing the C1 data, by default `None`. If `None`, any of data_directory,
            c2_directory, or c3_directory must be defined. This parameter is ignored if data_directory is defined.
        c2_directory: str, optional
            The path to the directory containing the C2 data, by default `None`. If `None`, any of data_directory,
            c1_directory, or c3_directory must be defined. This parameter is ignored if data_directory is defined.
        c3_directory: str, optional
            The path to the directory containing the C3 data, by default `None`. If `None`, any of data_directory,
            c1_directory, or c2_directory must be defined. This parameter is ignored if data_directory is defined.
        rebin_factor: int, optional
            The rebinning factor, by default 1 (no rebinning). The rebinning factor is the factor by which the image is
            rebinned in both dimensions. Rebinning can improve the detectability of faint sources and speed up
            some operations (like cosmic ray removal) at the cost of image resolution.
        flat_corrector: FlatFieldCorrector, optional,
            The flat-field corrector, by default `None`. If `None`, no flat-field corrections are applied.
        threshold: float, optional
            The signal-to-noise ratio threshold for source finding, by default 5. Reduce this value to identify fainter
            sources, though this may lead to the identification of spurious sources.
        background: BaseBackground | None, optional
            The background calculator, by default `None`. If `None`, the default background calculator is used. If a
            callable is provided, it should take an image (`NDArray`) as input and return a `Background2D` object.
        finder: Callable, optional
            The source finder, by default `None`. If `None`, the default source finder is used. If a callable is
            provided, it should take an image (`NDArray`) and a threshold (`float`) as input and return a
            `SegmentationImage` object.
        aperture_selector: Callable, optional
            The aperture selector, by default `np.median`. This function is used to select the aperture size for
            photometry. If a callable is provided, it should take a list of source sizes (`List[float]`) as input and
            return a single value.
        remove_cosmic_rays: bool, optional
            Whether to remove cosmic rays from images, by default False. Cosmic rays are removed using the LACosmic
            algorithm as implemented in `astroscrappy`. Note: this can be computationally expensive, particularly for
            large images.
        number_of_processors: int, optional
            The number of processors to use for parallel processing, by default half the number of available processors.
        show_plots: bool, optional
            Whether to show plots as they're created, by default `True`. Whether `True` or `False`, plots are always
            saved to `out_directory`.
        verbose: bool, optional
            Whether to print verbose output, by default `True`.
        """
        
        self.verbose = verbose
        
        ########################################### out_directory ###########################################
        
        self.out_directory = out_directory
        
        # create output directory if it does not exist
        if not os.path.isdir(self.out_directory):
            if self.verbose:
                print(f"[OPTICAM] {self.out_directory} not found, attempting to create ...")
            # create output directory if it does not exist
            try:
                os.makedirs(self.out_directory)
            except:
                raise FileNotFoundError(f"[OPTICAM] Could not create directory {self.out_directory}")
            if self.verbose:
                print(f"[OPTICAM] {self.out_directory} created.")
        
        ########################################### logger ###########################################
        
        # configure logger
        self.logger = logging.getLogger('OPTICAM')
        self.logger.setLevel(logging.INFO)
        
        # clear existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # create file handler
        file_handler = logging.FileHandler(os.path.join(self.out_directory, 'info.log'))
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        ########################################### sub-directories ###########################################
        
        # create subdirectories
        if not os.path.isdir(os.path.join(self.out_directory, "cat")):
            os.makedirs(os.path.join(self.out_directory, "cat"))
        if not os.path.isdir(os.path.join(self.out_directory, "diag")):
            os.makedirs(os.path.join(self.out_directory, "diag"))
        if not os.path.isdir(os.path.join(self.out_directory, "misc")):
            os.makedirs(os.path.join(self.out_directory, "misc"))
        
        ########################################### data directories ###########################################
        
        self.data_directory = data_directory
        self.c1_directory = c1_directory
        self.c2_directory = c2_directory
        self.c3_directory = c3_directory
        
        assert self.data_directory is not None or self.c1_directory is not None or self.c2_directory is not None or self.c3_directory is not None, "[OPTICAM] At least one of data_directory, c1_directory, c2_directory, or c3_directory must be defined."
        
        ########################################### input params ###########################################
        
        # set some useful parameters
        self.rebin_factor = rebin_factor
        self.flat_corrector = flat_corrector
        self.aperture_selector = aperture_selector
        self.threshold = threshold
        self.remove_cosmic_rays = remove_cosmic_rays
        self.number_of_processors = number_of_processors
        self.show_plots = show_plots
        
        ########################################### check input data ###########################################
        
        self.camera_files, self.binning_scale, self.bmjds, self.ignored_files, self.gains, self.t_ref = check_data(
                data_directory=data_directory,
                c1_directory=c1_directory,
                c2_directory=c2_directory,
                c3_directory=c3_directory,
                out_directory=out_directory,
                verbose=verbose,
                return_output=True,
                logger=self.logger,
                number_of_processors=number_of_processors,
                )  # type: ignore
        
        ########################################### plot time between files ###########################################
        
        plot_time_between_files(
            out_directory=self.out_directory,
            camera_files=self.camera_files,
            bmjds=self.bmjds,
            show=self.show_plots,
            save=True,
            )
        
        ########################################### define reference images ###########################################
        
        # define middle image as reference image for each filter
        reference_indices = {}
        self.reference_files = {}
        for key in self.camera_files.keys():
            reference_indices[key] = len(self.camera_files[key]) // 2
            self.reference_files[key] = self.camera_files[key][reference_indices[key]]
        
        ########################################### aperture selector ###########################################
        
        assert callable(aperture_selector), "[OPTICAM] aperture_selector must be callable."
        self.aperture_selector = aperture_selector
        
        ########################################### background ###########################################
        
        if background is None:
            box_size = 2048 // self.binning_scale // self.rebin_factor // 32
            self.background = DefaultBackground(box_size)
            self.logger.info(f'[OPTICAM] Using default background estimator with box_size={box_size}.')
        elif callable(background):
            # use custom background estimator
            self.background = background
            self.logger.info(f'[OPTICAM] Using custom background estimator {background.__class__.__name__} with parameters {background.__dict__}.')
        else:
            raise ValueError('[OPTICAM] background must be a callable or None. If None, the default background estimator is used.')
        
        ########################################### finder ###########################################
        
        if finder is None:
            effective_image_size = 2048 // self.binning_scale // self.rebin_factor
            npixels = 128 // (2048 // effective_image_size)**2
            border_width = 2048 // self.binning_scale // self.rebin_factor // 16
            self.finder = DefaultFinder(npixels, border_width)
            self.logger.info(f'[OPTICAM] Using default source finder with npixels={npixels} and border_width={border_width}.')
        elif callable(finder):
            self.finder = finder
            self.logger.info(f'[OPTICAM] Using custom source finder {finder.__class__.__name__} with parameters {finder.__dict__}.')
        else:
            raise ValueError('[OPTICAM] finder must be a callable or None. If None, the default source finder is used.')
        
        ########################################### log input params ###########################################
        
        log_reducer_params(self)
        
        ########################################### misc attributes ###########################################
        
        self.transforms = {}  # define transforms as empty dictionary
        self.unaligned_files = []  # define unaligned files as empty list
        self.catalogs = {}  # define catalogs as empty dictionary
        self.psf_params = {}  # define PSF parameters as empty dictionary
        
        ########################################### read transforms ###########################################
        
        if os.path.isfile(os.path.join(self.out_directory, "cat/transforms.json")):
            with open(os.path.join(self.out_directory, "cat/transforms.json"), "r") as file:
                self.transforms.update(json.load(file))
            
            if self.verbose:
                self.logger.info("[OPTICAM] Read transforms from file.")
        
        ########################################### read catalogs ###########################################
        
        for fltr in list(self.camera_files.keys()):
            if os.path.isfile(os.path.join(self.out_directory, f"cat/{fltr}_catalog.ecsv")):
                self.catalogs.update(
                    {
                        fltr: QTable.read(
                            os.path.join(self.out_directory, f"cat/{fltr}_catalog.ecsv"),
                            format="ascii.ecsv",
                            )
                        }
                    )
                self.psf_params[fltr] = set_psf_params(
                    fltr=fltr,
                    aperture_selector=self.aperture_selector,
                    out_directory=self.out_directory,
                    catalog=self.catalogs[fltr],
                    binning_scale=self.binning_scale,
                    rebin_factor=self.rebin_factor,
                )
                if self.verbose:
                    print(f"[OPTICAM] Read {fltr} catalog from file.")

    def create_catalogs(
        self,
        max_catalog_sources: int = 30,
        n_alignment_sources: int = 30,
        transform_type: Literal['affine', 'translation'] = 'affine',
        rotation_limit: float | None = None,
        translation_limit: float | int | List[float | int] | None = None,
        scale_limit: float | None = None,
        overwrite: bool = False,
        show_diagnostic_plots: bool = False,
        ) -> None:
        """
        Initialise the source catalogs for each camera. Some aspects of this method are parallelised for speed.
        
        Parameters
        ----------
        max_catalog_sources : int, optional
            The maximum number of sources to include in the catalog, by default 30. Since source IDs are ordered by
            brightness, the brightest `max_catalog_sources` sources are included in the catalog.
        n_alignment_sources : int, optional
            The (maximum) number of sources to use for image alignment, by default 30. If
            `transform_type='translation'`, `n_alignment_sources` must be >= 1, and the brightest `n_alignment_sources`
            sources are used for image alignment. If `transform_type='affine'`, `n_alignment_sources` must be >= 3 and
            represents that *maximum* number of sources that *may* be used for image alignment.
        transform_type : Literal['affine', 'translation'], optional
            The type of transform to use for image alignment, by default 'affine'. 'translation' performs simple
            x, y translations, while 'affine' uses `astroalign.find_transform()`. 'affine' is generally more robust 
            (and is therefore the default) while 'translation' can work with fewer sources.
        rotation_limit : float, optional
            The maximum rotation limit (in degrees) for affine transformations, by default `None` (no limit).
        scale_limit : float, optional
            The maximum scale limit for affine transformations, by default `None` (no limit).
        translation_limit : float | int | List[float | int] | None, optional
            The maximum translation limit for both types of transformations, by default `None` (no limit). Can be a
            scalar value that applies to both x- and y-translations, or an iterable where the first value defines the
            x-translation limit and the second value defines the y-translation limit.
        overwrite : bool, optional
            Whether to overwrite existing catalogs, by default False.
        show_diagnostic_plots : bool, optional
            Whether to show diagnostic plots, by default False. Diagnostic plots are saved to out_directory, so this
            parameter only affects whether the plots are displayed in the console.
        """
        
        assert transform_type in ['affine', 'translation'], '[OPTICAM] transform_type must be either "affine" or "translation".'
        
        if translation_limit is not None:
            # if a scalar translation limit is specified, convert it to a list
            if isinstance(translation_limit, float) or isinstance(translation_limit, int):
                translation_limit = [translation_limit, translation_limit]
        
        # if catalogs already exist, skip
        if os.path.isfile(os.path.join(self.out_directory, 'cat/catalogs.png')) and not overwrite:
            print('[OPTICAM] Catalogs already exist. To overwrite, set overwrite=True.')
            
            plot_catalogs(
                out_directory=self.out_directory,
                stacked_images=get_stacked_images(self.out_directory),
                catalogs=self.catalogs,
                show=self.show_plots,
                save=False,
            )
            
            return
        
        if self.verbose:
            print('[OPTICAM] Creating source catalogs')
        
        background_median = {}
        background_rms = {}
        stacked_images = {}
        
        instance_align_batch = partial(
            align_batch,
            gains=self.gains,
            flat_corrector=self.flat_corrector,
            rebin_factor=self.rebin_factor,
            remove_cosmic_rays=self.remove_cosmic_rays,
            background=self.background,
            finder=self.finder,
            threshold=self.threshold,
            logger=self.logger,
        )
        
        # for each camera
        for fltr in self.camera_files.keys():
            
            # if no images found for camera, skip
            if len(self.camera_files[fltr]) == 0:
                continue
            
            # get reference image
            # np.asarray() to fix type error
            reference_image = np.asarray(
                get_data(
                    file=self.reference_files[fltr],
                    gain=self.gains[self.reference_files[fltr]],
                    return_error=False,
                    flat_corrector=self.flat_corrector,
                    rebin_factor=self.rebin_factor,
                    remove_cosmic_rays=self.remove_cosmic_rays,
                    )
                )
            
            try:
                # get source coordinates in descending order of brightness
                reference_coords = get_source_coords_from_image(
                    reference_image,
                    finder=self.finder,
                    threshold=self.threshold,
                    background=self.background,
                    n_sources=n_alignment_sources,
                    )
            except Exception as e:
                self.logger.info(f'[OPTICAM] No sources detected in {fltr} reference image ({self.reference_files[fltr]}): {e}. Reducing threshold or npixels in the source finder may help.')
                continue
            
            if len(reference_coords) < n_alignment_sources and transform_type == 'translation':
                self.logger.info(f'[OPTICAM] Found {len(reference_coords)} sources in {fltr} reference image ({self.reference_files[fltr]}) but n_alignment_sources={n_alignment_sources}. transform_type="translation" requires at least n_alignment_sources be detected in the reference image to work. Consider reducing n_alignment_sources and/or threshold, or using transform_type="affine".')
                continue
            
            self.logger.info(f'[OPTICAM] {fltr} alignment source coordinates: {reference_coords}')
            
            # align and stack images in batches
            batches = get_batches(self.camera_files[fltr])
            results = process_map(
                partial(
                    instance_align_batch,
                    reference_image_shape=reference_image.shape,
                    reference_coords=reference_coords,
                    transform_type=transform_type,
                    rotation_limit=rotation_limit,
                    scale_limit=scale_limit,
                    translation_limit=translation_limit,
                    n_alignment_sources=n_alignment_sources,
                    ),
                batches,
                max_workers=self.number_of_processors,
                disable=not self.verbose,
                desc=f'[OPTICAM] Aligning {fltr} images',
                bar_format=bar_format,
                tqdm_class=tqdm,
                )
            
            self.transforms, self.unaligned_files, stacked_image, background_median[fltr], background_rms[fltr] = \
                parse_alignment_results(
                    results=results,
                    camera_files=self.camera_files[fltr],
                    transforms=self.transforms,
                    unaligned_files=self.unaligned_files,
                    verbose=self.verbose,
                    )
            
            try:
                # estimate threshold for source detection
                threshold = detect_threshold(
                    stacked_image,
                    nsigma=self.threshold,
                    )
            except Exception as e:
                self.logger.info(f'[OPTICAM] Unable to estimate source detection threshold for {fltr} stacked image: {e}.')
                continue
            
            try:
                # identify sources in stacked image
                segment_map = self.finder(
                    stacked_image,
                    threshold,
                    )
            except Exception as e:
                self.logger.info(f'[OPTICAM] No sources detected in the stacked {fltr} stacked image: {e}. Reducing threshold may help.')
                continue
            
            # save stacked image
            stacked_images[fltr] = stacked_image
            
            # catalog sources
            tbl = SourceCatalog(stacked_image, segment_map).to_table()
            tbl.sort('segment_flux', reverse=True)
            tbl = tbl[:max_catalog_sources]  # limit catalog to brightest max_catalog_sources sources
            
            # save catalog
            self.catalogs.update({fltr: tbl})
            self.catalogs[fltr].write(
                os.path.join(self.out_directory, f"cat/{fltr}_catalog.ecsv"),
                format="ascii.ecsv",
                overwrite=True,
                )
            
            self.psf_params[fltr] = set_psf_params(
                fltr=fltr,
                aperture_selector=self.aperture_selector,
                out_directory=self.out_directory,
                catalog=self.catalogs[fltr],
                binning_scale=self.binning_scale,
                rebin_factor=self.rebin_factor,
                )
        
        plot_catalogs(
            out_directory=self.out_directory,
            stacked_images=stacked_images,
            catalogs=self.catalogs,
            show=self.show_plots,
            save=True,
        )
        
        plot_backgrounds(
            out_directory=self.out_directory,
            camera_files=self.camera_files,
            background_median=background_median,
            background_rms=background_rms,
            bmjds=self.bmjds,
            t_ref=self.t_ref,
            show=show_diagnostic_plots,
            save=True,
        )
        
        plot_background_meshes(
            out_directory=self.out_directory,
            filters=list(self.camera_files.keys()),
            stacked_images=stacked_images,
            background=self.background,
            show=show_diagnostic_plots,
            save=True,
        )
        
        save_stacked_images(
            stacked_images=stacked_images,
            out_directory=self.out_directory,
            overwrite=overwrite,
            )
        
        # save transforms to file
        if not os.path.isfile(os.path.join(self.out_directory, "cat/transforms.json")) or overwrite:
            with open(os.path.join(self.out_directory, "cat/transforms.json"), "w") as file:
                json.dump(self.transforms, file, indent=4)
        
        # write unaligned files to file
        if len(self.unaligned_files) > 0 and (not os.path.isfile(os.path.join(self.out_directory, "diag/unaligned_files.txt")) or overwrite):
            with open(os.path.join(self.out_directory, "diag/unaligned_files.txt"), "w") as unaligned_file:
                for file in self.unaligned_files:
                    unaligned_file.write(file + "\n")

    def plot_psfs(
        self,
        ) -> None:
        
        if not os.path.isdir(os.path.join(self.out_directory, 'psfs')):
            os.makedirs(os.path.join(self.out_directory, 'psfs'))
        
        # get stacked images
        stacked_images = get_stacked_images(self.out_directory)
        
        for fltr in self.catalogs.keys():
            x_lo, x_hi = 0, stacked_images[fltr].shape[1]
            y_lo, y_hi = 0, stacked_images[fltr].shape[0]
            
            for source in tqdm(
                self.catalogs[fltr]['label'],
                disable=not self.verbose,
                desc=f'[OPTICAM] Plotting {fltr} PSFs',
                bar_format=bar_format,
                ):
                
                x, y = int(self.catalogs[fltr]['xcentroid'][source - 1]), int(self.catalogs[fltr]['ycentroid'][source - 1])  # source position
                w = int(self.catalogs[fltr]['semimajor_sigma'][source - 1].value) * 10  # source stdev
                x_range = np.arange(max(x_lo, int(x - w)), min(x_hi, int(x + w)))  # x range
                y_range = np.arange(max(y_lo, int(y - w)), min(y_hi, int(y + w)))  # y range
                
                # create mask
                mask = np.zeros_like(stacked_images[fltr], dtype=bool)
                for x in x_range:
                    for y in y_range:
                        mask[y, x] = True
                
                # isolate source
                rows_to_keep = np.any(mask, axis=1)
                region = stacked_images[fltr][rows_to_keep, :]
                cols_to_keep = np.any(mask, axis=0)
                region = region[:, cols_to_keep]
                
                fig = plt.figure(num=1, clear=True)
                ax = fig.add_subplot(projection='3d')
                
                x, y = np.meshgrid(x_range, y_range)
                
                ax.plot_surface(x, y, region, edgecolor='r', rstride=2, cstride=2, color='none', lw=.5)
                # ax.contour(x, y, region, 20, zdir='x', offset=ax.set_xlim()[0], colors='black', linewidths=.5)
                # ax.contour(x, y, region, 20, zdir='y', offset=ax.set_ylim()[1], colors='black', linewidths=.5)
                ax.contour(x, y, region, 10, zdir='z', offset=ax.set_zlim()[0], colors='black', linewidths=.5)
                
                ax.set_title(f'{fltr} source {source}')
                
                def update(frame):
                    ax.view_init(elev=30, azim=frame)
                    return fig,
                
                ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 5), interval=100, blit=True)
                ani.save(
                    os.path.join(
                        self.out_directory,
                        f'psfs/{fltr}_source_{source}.gif'),
                    writer='pillow',
                    fps=30,
                    )
                plt.close(fig)

    def create_gifs(self, keep_frames: bool = True, overwrite: bool = False) -> None:
        """
        Create alignment gifs for each camera. Some aspects of this method are parallelised for speed. The frames are 
        saved in out_directory/diag/*-band_gif_frames and the GIFs are saved in out_directory/cat.
        
        Parameters
        ----------
        keep_frames : bool, optional
            Whether to save the GIF frames in out_directory/diag, by default True. If False, the frames will be deleted
            after the GIF is saved.
        overwrite : bool, optional
            Whether to overwrite existing GIFs, by default False.
        """
        
        # for each camera
        for fltr in list(self.catalogs.keys()):
            
            # skip cameras with no images
            if len(self.camera_files[fltr]) == 0:
                continue
            elif os.path.exists(os.path.join(self.out_directory, f"cat/{fltr}_images.gif")) and not overwrite:
                print(f"[OPTICAM] {fltr} GIF already exists. To overwrite, set overwrite to True.")
                continue
            
            # create gif frames directory if it does not exist
            if not os.path.isdir(os.path.join(self.out_directory, f"diag/{fltr}_gif_frames")):
                os.mkdir(os.path.join(self.out_directory, f"diag/{fltr}_gif_frames"))
            
            chunksize = get_batch_size(len(self.camera_files[fltr]))
            process_map(
                partial(
                    create_gif_frame,
                    out_directory=self.out_directory,
                    aperture_selector=self.aperture_selector,
                    catalog=self.catalogs[fltr],
                    fltr=fltr,
                    gains=self.gains,
                    transforms=self.transforms,
                    reference_file=self.reference_files[fltr],
                    flat_corrector=self.flat_corrector,
                    rebin_factor=self.rebin_factor,
                    remove_cosmic_rays=self.remove_cosmic_rays,
                    background=self.background,
                    ),
                self.camera_files[fltr],
                max_workers=self.number_of_processors,
                disable=not self.verbose,
                desc=f"[OPTICAM] Creating {fltr} GIF frames",
                chunksize=chunksize,
                bar_format=bar_format,
                tqdm_class=tqdm,
                )
            
            # save GIF
            compile_gif(
                out_directory=self.out_directory,
                fltr=fltr,
                camera_files=self.camera_files,
                keep_frames=keep_frames,
                verbose=self.verbose,
                )

    def photometry(
        self,
        photometer: BasePhotometer,
        overwrite: bool = False,
        ) -> None:
        """
        Perform photometry on the catalogs using the provided photometer.
        
        Parameters
        ----------
        photometer : BasePhotometer
            The photometer. Should be a subclass of `BasePhotometer`, or implement a `compute` method that follows the
            `BasePhotometer` interface.
        overwrite : bool, optional
            Whether to overwrite any existing light curves files computed using the same photometer, by default `False`.
        """
        
        # define save directory using the photometer name
        save_name = camel_to_snake(photometer.__class__.__name__).replace('_photometer', '')
        
        # change save directory based on photometer settings
        if photometer.local_background_estimator is not None:
            save_name += '_annulus'
        if photometer.forced:
            save_name = 'forced_' + save_name
        
        print(f'[OPTICAM] Photometry results will be saved to {save_name}_light_curves in {self.out_directory}.')
        
        save_dir = os.path.join(self.out_directory, f"{save_name}_light_curves")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        # for each filter
        for fltr in self.catalogs.keys():
            
            if os.path.isfile(os.path.join(self.out_directory, f'{save_name}_light_curves/{fltr}_source_1.csv')) and not overwrite:
                print(f'[OPTICAM] Skipping {fltr} since existing light curves files were found. To overwrite these files, set overwrite=True.')
                continue
            
            source_coords = np.array([self.catalogs[fltr]["xcentroid"].value,
                                      self.catalogs[fltr]["ycentroid"].value]).T
            
            batch_size = get_batch_size(len(self.camera_files[fltr]))
            results = process_map(
                partial(
                    perform_photometry,
                    photometer=photometer,
                    source_coords=source_coords,
                    gains=self.gains,
                    bmjds=self.bmjds,
                    flat_corrector=self.flat_corrector,
                    rebin_factor=self.rebin_factor,
                    remove_cosmic_rays=self.remove_cosmic_rays,
                    background=self.background,
                    threshold=self.threshold,
                    finder=self.finder,
                    psf_params=self.psf_params,
                    fltr=fltr,
                    logger=self.logger,
                ),
                self.camera_files[fltr],
                max_workers=self.number_of_processors,
                disable=not self.verbose,
                desc=f"[OPTICAM] Performing photometry on {fltr} images",
                chunksize=batch_size,
                bar_format=bar_format,
                tqdm_class=tqdm,
            )
            
            save_photometry_results(
                results=results,
                catalogs=self.catalogs,
                save_dir=save_dir,
                fltr=fltr
            )




################### for a clearner UI, the following functions are intentionally not Catalog methods ###################

def log_reducer_params(
    reducer: Reducer,
    ) -> None:
    
    # get parameters
    params = dict(recursive_log(reducer, max_depth=5))
    
    params.update({'filters': list(reducer.camera_files.keys())})
    
    # remove some parameters that are either already saved elsewhere or are not relevant
    params.pop('logger')
    params.pop('bmjds')
    params.pop('gains')
    params.pop('camera_files')
    
    try:
        params.pop('transforms')
    except KeyError:
        pass
    
    try:
        params.pop('unaligned_files')
    except KeyError:
        pass
    
    try:
        params.pop('catalogs')
    except KeyError:
        pass
    
    # sort parameters
    params = dict(sorted(params.items()))
    
    # write parameters to file
    with open(os.path.join(reducer.out_directory, "misc/reduction_parameters.json"), "w") as file:
        json.dump(params, file, indent=4)


def set_psf_params(
    fltr: str,
    out_directory: str,
    aperture_selector: Callable,
    catalog: QTable,
    binning_scale: int,
    rebin_factor: int,
    ) -> Dict[str, float]:
    """
    Set the PSF parameters for a given filter based on the catalog data.
    
    Parameters
    ----------
    fltr : str
        The filter for which to set the PSF parameters.
    """
    
    
    semimajor_sigma_pix = aperture_selector(catalog['semimajor_sigma'].value)
    semiminor_sigma_pix = aperture_selector(catalog['semiminor_sigma'].value)
    orientation = aperture_selector(catalog['orientation'].value)
    
    semimajor_sigma_arcsec = semimajor_sigma_pix * binning_scale * rebin_factor * pixel_scales[fltr]
    semiminor_sigma_arcsec = semiminor_sigma_pix * binning_scale * rebin_factor * pixel_scales[fltr]
    
    # PSF params used by Catalog (pixels only)
    psf_params_pix = {
        'semimajor_sigma': semimajor_sigma_pix,
        'semiminor_sigma': semiminor_sigma_pix,
        'orientation': orientation,
    }
    
    psf_params_full = {
        'semimajor_sigma_arcsec': semimajor_sigma_arcsec,
        'semimajor_sigma_pix': semimajor_sigma_pix,
        'semiminor_sigma_arcsec': semiminor_sigma_arcsec,
        'semiminor_sigma_pix': semiminor_sigma_pix,
        'orientation': orientation,
    }
    
    # save PSF params to JSON file
    with open(os.path.join(out_directory, f'misc/psf_params.json'), 'w') as file:
        json.dump(psf_params_full, file, indent=4)
    
    return psf_params_pix


def parse_alignment_results(
    results: Tuple,
    camera_files: List[str],
    transforms: Dict[str, List[float]],
    unaligned_files: List[str],
    verbose: bool,
    ):
    
    fltr_transforms = {}
    fltr_unaligned_files = []
    fltr_background_medians = {}
    fltr_background_rmss = {}
    
    # unpack results
    batch_stacked_images, batch_transforms, batch_background_medians, batch_background_rmss = zip(*results)
    
    # combine results
    for i in range(len(batch_stacked_images)):
        fltr_transforms.update(batch_transforms[i])
        fltr_background_medians.update(batch_background_medians[i])
        fltr_background_rmss.update(batch_background_rmss[i])
    
    aligned_files = list(fltr_transforms.keys())
    for file in camera_files:
        if file not in aligned_files:
            fltr_unaligned_files.append(file)
    
    stacked_image = np.sum(batch_stacked_images, axis=0)  # stack images
    
    transforms.update(fltr_transforms)  # update transforms to include current filter
    unaligned_files += fltr_unaligned_files  # update unaligned files
    
    if verbose:
        print(f"[OPTICAM] Done.")
        print(f'[OPTICAM] {len(fltr_transforms)} image(s) aligned.')
        print(f'[OPTICAM] {len(fltr_unaligned_files)} image(s) could not be aligned.')
    
    return transforms, unaligned_files, stacked_image, fltr_background_medians, fltr_background_rmss


def save_photometry_results(
    results: Tuple[Dict],
    catalogs: Dict[str, QTable],
    save_dir: str,
    fltr: str,
    ):
    """
    Save the photometry results to disk.
    
    Parameters
    ----------
    results : Tuple[Dict]
        The photometry results.
    catalogs : Dict[str, QTable]
        The source catalogs.
    save_dir : str
        The save directory path.
    fltr : str
        The photometry filter.
    """
    
    photometry_results = parse_photometry_results(results)
    
    # for each source in the catalog
    for i in range(len(catalogs[fltr])):
        
        # unpack results for ith source
        source_results = {}
        for key, values in photometry_results.items():
            
            # time is a special case since it is already a single column
            if key == 'BMJD':
                source_results[key] = np.asarray(values)
            # for other keys, the ith column needs to be extracted
            else:
                col = [value[i] for value in values]
                source_results[key] = np.asarray(col)
        
        # define light curve as a DataFrame
        df = pd.DataFrame(source_results)
        
        # drop NaNs
        df.dropna(inplace=True, ignore_index=True)
        df.reset_index(drop=True, inplace=True)
        
        # save to file
        df.to_csv(
            os.path.join(
                save_dir,
                f'{fltr}_source_{i + 1}.csv',
                ),
            index=False,
            )

def parse_photometry_results(
    results: Tuple[Dict[str, List]],
    ) -> Dict[str, List[List[float]]]:
    """
    Merge the multiprocessed photometry results into a single dictionary.
    
    Parameters
    ----------
    results : Tuple[Dict[str, List]]
        The multiprocessed photometry results.
    
    Returns
    -------
    Dict[str, List[List[float]]]
        The photometry results in a single dictionary.
    """
    
    photometry_results = {}
    for result in results:
        for key, value in result.items():
            if key not in photometry_results:
                photometry_results[key] = []
            photometry_results[key].append(value)
    
    return photometry_results














