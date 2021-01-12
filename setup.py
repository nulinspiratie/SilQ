from setuptools import setup, find_packages
from distutils.version import StrictVersion
from importlib import import_module
import re

def get_version(verbose=1):
    """ Extract version information from source code """

    try:
        with open('silq/version.py', 'r') as f:
            ln = f.readline()
            # print(ln)
            m = re.search('.* ''(.*)''', ln)
            version = (m.group(1)).strip('\'')
    except Exception as E:
        print(E)
        version = 'none'
    if verbose:
        print('get_version: %s' % version)
    return version


def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()

extras = {
    'Non-Linear Least-Squares Minimization and Curve-Fitting': ('lmfit', '0.9.5'),
    'Peak finding': ('peakutils', '1.0.3'),
    'PythonNet': ('pythonnet', '2.1'),
    'slacker': ('slacker', '0.9.42'),
    'theano': ('theano', '0.9'),
    'keras': ('keras', '2.0.8'),
    'qutip': ('qutip', '4.5.0')
    # 'ipympl': ('ipympl', '0.0.3')

}
extras_require = {k: '>='.join(v) for k, v in extras.items()}

setup(name='silq',
      version=get_version(),
      use_2to3=False,
      author='Serwan Asaad, Mark Johnson, Hannes Firgau, Stefanie Tenberg, Rostyslav Savytskyy, Irene Fernandez de Fuentes',
      author_email='serwan.asaad@gmail.com',
      maintainer='Serwan Asaad, Stefanie Tenberg',
      maintainer_email='serwan.asaad@gmail.com',
      description='Python-based measurement framework for qubit measurements developed'
                  ' by the research group of Andrea Morello in Sydney',
      long_description=readme(),
      url='https://github.com/nulinspiratie/SilQ',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3.5',
          'Topic :: Scientific/Engineering'
      ],
      license='Private',
      packages=find_packages(),
      package_data={'silq': ['widgets/*.js', 'widgets/*.css']},
      install_requires=[
          'numpy>=1.10',
          'pyvisa>=1.8',
          'IPython>=4.0',
          'ipywidgets>=4.1',
          'h5py>=2.6',
          'pyperclip>=1.5',
          'blinker',
          'dill',
          'regex'
      ],
      extras_require=extras_require,
      zip_safe=False)

version_template = '''
***** package {0} must be at least version {1}.
***** Please upgrade it (pip install -U {0} or conda install {0})
***** in order to use {2}
'''

missing_template = '''
***** package {0} not found
***** Please install it (pip install {0} or conda install {0})
***** in order to use {1}
'''

valueerror_template = '''
***** package {0} version not understood
***** Please make sure the installed version ({1})
***** is compatible with the minimum required version ({2})
***** in order to use {3}
'''

# now test the versions of extras
for extra, (module_name, min_version) in extras.items():
    try:
        module = import_module(module_name)
        if StrictVersion(module.__version__) < StrictVersion(min_version):
            print(version_template.format(module_name, min_version, extra))
    except ImportError:
        print(missing_template.format(module_name, extra))
    except ValueError:
        print(valueerror_template.format(
            module_name, module.__version__, min_version, extra))
