import os
import sys

a = input("have you updated conf.py, __init__.py and setup.py with the version"
          "name? (y/n)")
if a == 'yes':
    os.system('cmd /k "python setup.py sdist bdist_wheel"')
else:
    sys.exit()

b = input('Check it - all OK? (y/n)')
if b == 'yes':
    os.system('cmd /k "twine upload --repository-url https://test.pypi.org/'
              'legacy/ dist/*"')
