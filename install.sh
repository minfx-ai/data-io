sudo apt update
sudo apt install build-essential autoconf libtool pkg-config
sudo apt install libpsl-dev
cd curl
./buildconf     # Generate the configure script
./configure --with-openssl    # Configure build options (add --prefix=/your/path to install elsewhere)
make -j $(nproc)  #Build libcurl


 echo "$USER           hard    memlock         1048576" >> /etc/security/limits.conf
 echo "$USER           soft    memlock         1048576" >> /etc/security/limits.conf
 echo "session required pam_limits.so" >> /etc/pam.d/common-session


 grep Hugepagesize /proc/meminfo
 echo 256 > /proc/sys/vm/nr_hugepages