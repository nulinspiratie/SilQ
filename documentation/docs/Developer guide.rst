===============
Developer guide
===============




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