https://github.com/h5py/h5py/releases/tag/2.9.0
https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.11/obtain51811.html
https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=2ahUKEwiopojo4LXmAhWIAhAIHY6FDmkQFjAAegQIARAB&url=https%3A%2F%2Fbootstrap.pypa.io%2Fget-pip.py&usg=AOvVaw0zKVO_zW0nkF7s0zdjWFNj

/opt/hfs18.0.287/python/bin/python get-pip.py
udo chown pawel -R /opt/hfs18.0.287/python/
/opt/hfs18.0.287/python/bin/python -m pip install pkgconfig
# Due to https://github.com/keras-team/keras/issues/13353
/opt/hfs18.0.287/python/bin/python -m pip install keras==2.2.5
/opt/hfs18.0.287/python/bin/python -m pip install tensorflow==1.15

/opt/hfs18.0.287/python/bin/python setup.py configure --hdf5-version=1.8.11 --hdf5=/home/pawel/git/hdf5-1.8.11-linux-x86_64-shared
/opt/hfs18.0.287/python/bin/python setup.py install