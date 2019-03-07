===============
Developer guide
===============


Adding a new feature/fixing a bug
---------------------------------
SilQ is meant to be a collaborative software, and so users are encouraged to
contribute any features and fixes for encountered bugs.
To submit any new features/bugfixes, a `Pull Request <https://help.github
.com/en/articles/creating-a-pull-request>` should be created.
This creates a proposed enhancement that can be `pulled` into the master branch.
The general procedure is as follows:

1. Start from the ``master`` branch without any modifications
2. Create a new branch, called ``feature/{branch_name}``, or
   ``fix/{branch_name}``, where ``{branch_name}`` should be a short clear name.
3. Implement the changes.
4. Push the changes to GitHub.
5. Create a Pull Request from your branch to the ``master`` branch.
   This can be done on the GitHub website.
   Be sure to give a clear description of the contents of the Pull Request

Once your pull request is submitted, other developers can review the proposed
changes, and accept/reject/ask for modifications.


Updating the GitHub documentation website
-----------------------------------------
The documentation for SilQ is hosted on GitHub, and uses the source code in
the SilQ ``gh-pages`` branch.
To upload any documentation changes to the website, follow these steps.

1. Go to the ``master`` branch
2. In a terminal, navigate to ``SilQ/documentation``
3. Execute ``make html``, which should create a folder ``SilQ-documentation``
   next the the main SilQ folder containing the website. The file
   ``SilQ-documentation/html/index.html`` is the root webpage.
4. Execute ``make gh-pages``. This command will switch to the ``gh-pages``
   branch, copy all the html code from ``SilQ-documentation``, commit and push
   all the changes, and return to the ``master`` branch.

After these steps, the website should have updated.