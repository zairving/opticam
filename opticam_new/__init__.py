
from opticam_new.background.global_background import BaseBackground, DefaultBackground
from opticam_new.background.local_background import BaseLocalBackground, DefaultLocalBackground
from opticam_new.finders import DefaultFinder
from opticam_new.catalog import Catalog
from opticam_new.analysis.differential_photometer import DifferentialPhotometer
from opticam_new.photometers import SimplePhotometer, OptimalPhotometer
from opticam_new.correctors.flat_field_corrector import FlatFieldCorrector
from opticam_new.analysis.analyzer import Analyzer
from opticam_new.utils.generate import generate_flats, generate_observations, generate_gappy_observations
from opticam_new.utils.data_checks import check_data

import warnings
warnings.warn('[OPTICAM] from version 0.3.0, `opticam_new` will be renamed to `opticam`.')