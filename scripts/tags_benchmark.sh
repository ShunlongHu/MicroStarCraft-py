commit="benchmark_$(git rev-parse --short HEAD)"
host="host_$(hostname)"
branch="branch_$(git rev-parse --abbrev-ref HEAD)"
version="v$(grep Version | sed -e 's#Version:\ ##')"
echo "$commit $host $branch $version"