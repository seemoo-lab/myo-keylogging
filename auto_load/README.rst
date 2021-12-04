auto-load
=========

This library can be used to load Python scripts from folders, automatically deriving a list of passable arguments together with a matching help message.
Internally, the argparse library is used to create subparsers for each scrip which contains a ``main`` function and is located inside a folder containing a ``__main__.py`` file with the following content:

.. code-block:: python

    # Note: Place this code in a __main__.py file inside each folder you want to execute scripts from.
    import auto_load
    if __name__ == "__main__":
        auto_load.execute(__file__, __package__)

The ``main`` function in each script serves as an entry point for running these scripts, enabling to pass arguments from the command line.
For each script, the positional and optional arguments are automatically derived by the parameters of the respective ``main`` function, as outlined in the following example.

.. code-block:: python

    # Note: Top level Docstring and function annotations are used to generate a help message.
    """
    This is a minimal working example for using the auto-load library.
    """
    def main(arg1: "description of arg1" = "default_value1"):
        print(arg1)

When saving the code from above to ``$FOLDERNAME/$SCRIPT_BASENAME.py``, one can then execute this script with the following command:

.. code-block:: bash

    user@computer> python -m $FOLDERNAME $SCRIPT_BASENAME --arg1 "Hello World"
    Hello World

For each script a help message is automatically derived from the top level Docstring, as well as from the function annotations and default values of the ``main`` function.

.. code-block:: bash

    user@computer> python -m $FOLDERNAME $SCRIPT_BASENAME --help
    usage: python -m $FOLDERNAME $SCRIPT_BASENAME [-h] [--arg1 ARG1]
    
    This is a minimal working example for using the auto-load library.
    
    optional arguments:
      -h, --help   show this help message and exit
      --arg1 ARG1  description of arg1 (default: default_value1)

Similarly, one can also get an overview of all scripts contained within a folder, given they contain a ``main`` function.

.. code-block:: bash

    user@computer> python -m $FOLDERNAME --help
    usage: python -m $FOLDERNAME [-h] {$SCRIPT_BASENAME,$OTHER_SCRIPT_BASENAME,...} ...
    
    positional arguments:
      {$SCRIPT_BASENAME,$OTHER_SCRIPT_BASENAME,...}
        $SCRIPT_BASENAME        This is a minimal working example for using the auto-load library.
        $OTHER_SCRIPT_BASENAME  ...
        ...
    
    optional arguments:
      -h, --help  show this help message and exit

License
-------

This library is licensed under the GPLv3 license.
