# Making contributions
* [Thank you!](#thank-you)
* [Whom to contact in case of questions?](#contact)
* [Submit a bug report or a feature request](#bug-feature)
* [How to contribute](#how-to-contribute)
    * [Code Requirements](#code-requirements)
    * [Necessary steps](#steps)
    * [Pull request Conventions](#pr)

# <a name="thank-you">Thank You!</a>
Before we get to the fine print and nitty gritty of contributing, let us start by saying *Thank you!* for considering contributing to the tuPlan Garage.
Together we can build a powerful plugin that allows fast benchmarking with comparable results across research.

# <a name="license">License</a>
* be aware that your code will be under Apache 2.0 license
* don't use third-party code that does not allow modification, distribution, usage, etc.

# <a name="contact">Whom to contact in case of questions?</a>
You find this repository's maintainers and contact authors in the `README.md`.

If you have questions regarding a specific model published in the tuPlan Garage, please consider contacting the code author directly.

If you have general questions regarding this repository or run into problems with the code, feel free to open an issue.

# <a name="bug-feature">Submit a bug report or a feature request</a>
Found a bug? Great! Well, not really, but at least we can fix it now. The best way to report a bug is to open an issue for this repository. Thank you for helping to improve our code!

# <a name="how-to-contribute">How to contribute</a>
If you added a new model to the tuPlan Garage or have fixed a bug or developed that new feature you would like to make available to your fellow users, we'd like to encourage you to contribute that to our codebase. In the sections below, you will find some requirements and guidelines for your contributions.

## <a name="code-requirements">Code Requirements</a>
To maintain high code quality, we need you to stick to the following coding conventions.
* Take special care that your code is reusable and easy to understand, modify, and test.
* tuPlan Garage is built to be a separate plugin for the nuPlan-framework. That means it should be compatible with the nuPlan-devkit without any modifications to the latter.
* Please check if you can reuse existing code before adding. This applies especially to features/targets types and utils functions. If a utils function needs refactoring to make it reusable for additional models, please open an issue and consider creating a pull request.
* To ensure seamless installation, remember to add additional required packages to the requirements file.
* If your model / feature-builders etc., consist of multiple files, please keep them in a separate folder within the `model/feature-builder` directory.
* To make your code readable and easy to understand, we encourage you to add docstrings and comments wherever necessary.
* Use the pre-commit hooks to ensure your code is formatted properly and checked for runaway errors such as unused imports or variables. Just install the hooks by running `pre-commit install` inside the repository. Afterward, your code will be checked and formatted properly before committing.

## <a name="steps">Necessary Steps</a>
This plugin's purpose is to ease fast and comparable benchmarking of state-of-the-art planners in nuPlan.
To ensure your model achieves reproducible results, we ask you for the following when contributing.
* Add the model code (Make sure to see [Code Requirements](#code-requirements))
* Add a training and simulation script in the `scripts` folder
* Provide a checkpoint that was generated with this model code and the respective script
* Add the results you achieved on the Val14 Benchmark with this checkpoint and your validation script to the table in the `README.md`. Ideally, also link the respective paper in the table.

**Note:** The repository maintainers will use your checkpoint and evaluation script to validate your results before accepting your pull request.

## <a name="pr">Pull Request Conventions</a>
* Pull Requests can only be merged after careful review by the package maintainers.
Please be patient and provide them with all the necessary information for a fast review.
* Write a good description to allow the reviewer to quickly get an overview of your changes.
