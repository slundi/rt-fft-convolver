Markdown

# How to Contribute to this Project

Thank you for your interest in contributing to this project! Your help is invaluable in making it better.

Please follow the guidelines below to submit your contributions effectively.

## Submitting Your Changes via Merge Requests

For any changes you wish to make to the project (bug fixes, feature additions, documentation improvements, etc.), please follow the **Merge Request** process (or Pull Request if you are using another source code management tool like GitHub).

Here are the steps to follow:

1.  **Fork the repository (if you are not a direct collaborator).** Create a copy of the repository on your own account.
2.  **Create a dedicated branch for your modification.** Give your branch a clear and descriptive name of the feature or fix you are implementing (e.g., `fix-login-bug`, `feat-new-feature`).
    ```bash
    git checkout -b my-descriptive-branch
    ```
3.  **Make your changes and commit them.** Ensure you follow the Conventional Commits convention (see the section below).
4.  **Push your branch to your fork (or the main repository if you are a collaborator).**
    ```bash
    git push origin my-descriptive-branch
    ```
5.  **Create a Merge Request (or Pull Request).** From your source code management interface (e.g., GitLab, GitHub), create a new Merge Request.
    * **Source Branch:** Select the branch you just pushed.
    * **Target Branch:** Select the main branch of the project (usually `main` or `master`).
    * **Title:** Give your Merge Request a clear and concise title that summarizes your change. Also, use the Conventional Commits convention for the title.
    * **Description:** Provide a detailed description of your modification. Explain the problem you are solving, the feature you are adding, or the improvement you are making. Include steps to test your changes if necessary.
    * **Assign a reviewer (if applicable).**
    * **Add labels (if applicable).**

The project team will review your Merge Request, may ask for clarifications or modifications, and will merge it once it is approved.

## Using the Conventional Commits Convention

We adhere to the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification to maintain a clear and structured commit history. This makes it easier to understand changes and automate certain processes (like generating changelogs).

A commit should be structured as follows:
