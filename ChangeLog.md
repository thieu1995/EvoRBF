
# Version 2.1.0

+ Updated the `setup` file.  
+ Updated `mealpy` dependency to version **3.0.2**.  
+ Updated `citation.cff` and `MANIFEST.in` files.  
+ Improved documentation, examples, and test cases.  
+ Updated the following modules: `metrics`, `preprocessor`, and `scaler`.  
+ Fixed a bug in `get_optimizer_by_class` within the `base_rbf` and `rbf_tuner` modules.  
+ Renamed the parameter `optim_paras` to `optim_params`.  
+ Removed the `obj_weights` parameter from `NiaRbfRegressor`.  
+ Added new parameters to the following classes: `NiaRbfRegressor`, `NiaRbfClassifier`, and `NiaRbfTuner`:  
  - `lb`, `ub`, `mode`, `n_workers`, and `termination`.  
+ Fixed missing `classes_` attribute in `RbfClassifier`, `NiaRbfClassifier`, and `AdvancedRbfClassifier`.  
+ Added **pretty-printing** support for all classes.

---------------------------------------------------------------------

# Version 2.0.0

+ Rename the title of framework
+ Rename the parameter for `score`, `scores` function.
+ Fix bug OneHotEncoder class
+ Update `standard_rbf` module with: `RbfRegressor` and `RbfClassifier` classes.
+ Add `advanced_rbf` module with: `AdvancedRbfNet`, `BaseAdvancedRbf`, `AdvancedRbfRegressor`, and `AdvancedRbfClassifier` classes
+ Update `nia_rbf` module with: `NiaRbfRegressor` and `NiaRbfClassifier` for regression and classification problems.
+ Update `rbf_tuner` module with: `NiaRbfTuner` class.
+ Add `center_finder` and `kernel` modules for `advanced_rbf` module.
+ Update documentation, examples, test cases.
+ Update citation, changelog, readme.

---------------------------------------------------------------------

# Version 1.0.0

+ Add InaRbfTuner class
+ Update docs for all classes
+ Update examples and documents
+ Add ChangeLog.md

---------------------------------------------------------------------


# Version 0.2.0 

+ Add NiaRbfClassifier class
+ Add traditional RBF model: RbfRegressor and RbfClassifier
+ Update examples, tests and docs
+ Add Github workflow


---------------------------------------------------------------------

# Version 0.1.0 (First version)

+ Add infors (CODE_OF_CONDUCT.md, MANIFEST.in, LICENSE, README.md, requirements.txt, CITATION.cff)
+ Add helpers modules (`metrics`, `scaler`, `validator`, and `preprocessor`)
+ Add NiaRbfRegressor class
+ Add examples and tests folders
