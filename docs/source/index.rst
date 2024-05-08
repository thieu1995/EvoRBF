.. EvoRBF documentation master file, created by
   sphinx-quickstart on Sat May 20 16:59:33 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EvoRBF's documentation!
==================================

.. image:: https://img.shields.io/badge/release-0.1.0-yellow.svg
   :target: https://github.com/thieu1995/evorbf/releases

.. image:: https://img.shields.io/pypi/wheel/gensim.svg
   :target: https://pypi.python.org/pypi/evorbf

.. image:: https://badge.fury.io/py/evorbf.svg
   :target: https://badge.fury.io/py/evorbf

.. image:: https://img.shields.io/pypi/pyversions/evorbf.svg
   :target: https://www.python.org/

.. image:: https://img.shields.io/pypi/status/evorbf.svg
   :target: https://img.shields.io/pypi/status/evorbf.svg

.. image:: https://img.shields.io/pypi/dm/evorbf.svg
   :target: https://img.shields.io/pypi/dm/evorbf.svg

.. image:: https://github.com/thieu1995/evorbf/actions/workflows/publish-package.yaml/badge.svg
   :target: https://github.com/thieu1995/evorbf/actions/workflows/publish-package.yaml

.. image:: https://static.pepy.tech/badge/evorbf
   :target: https://pepy.tech/project/evorbf

.. image:: https://img.shields.io/github/release-date/thieu1995/evorbf.svg
   :target: https://img.shields.io/github/release-date/thieu1995/evorbf.svg

.. image:: https://readthedocs.org/projects/evorbf/badge/?version=latest
   :target: https://evorbf.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/Chat-on%20Telegram-blue
   :target: https://t.me/+fRVCJGuGJg1mNDg1

.. image:: https://img.shields.io/github/contributors/thieu1995/evorbf.svg
   :target: https://img.shields.io/github/contributors/thieu1995/evorbf.svg

.. image:: https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?
   :target: https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1113600.svg
   :target: https://doi.org/10.5281/zenodo.1113600

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0


**EvoRBF: Evolving Radial Basis Function Network by Intelligent Nature-inspired Algorithms**

EvoRBF (Evolving Radial Basis Function Network) is a Python library that implements a framework
for training Radial Basis Function (RBF) networks using `Intelligence Nature-inspired Algorithms (INAs)`. It provides a
comparable alternative to the traditional RBF network and is compatible with the Scikit-Learn library. With EvoRBF, you can
perform searches and hyperparameter tuning using the functionalities provided by the Scikit-Learn library.

* **Free software:** GNU General Public License (GPL) V3 license
* **Provided Estimator**: RbfRegressor, RbfClassifier, InaRbfRegressor, InaRbfClassifier
* **Total InaRBf models**: > 400 Models
* **Supported performance metrics**: >= 67 (47 regressions and 20 classifications)
* **Supported loss functions**: >= 61 (45 regressions and 16 classifications)
* **Documentation:** https://evorbf.readthedocs.io/en/latest/
* **Python versions:** >= 3.8.x
* **Dependencies:** numpy, scipy, scikit-learn, pandas, mealpy, permetrics


.. toctree::
   :maxdepth: 4
   :caption: Quick Start:

   pages/quick_start.rst

.. toctree::
   :maxdepth: 4
   :caption: Models API:

   pages/evorbf.rst

.. toctree::
   :maxdepth: 4
   :caption: Support:

   pages/support.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
