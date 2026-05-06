# wpi-update-debs
生成大部分用于wpi-update的包

main与arm64文件夹内存放了一些包，有些包存放在其他仓库中请查看 [package.list](./package.list) 

## 使用方式
```shell

# 拉取存放在其他github仓库中的包
./update_repos.sh

# 构建所有包
./build.sh

```
会在当前项目路径下创建一个 output 文件夹，所有包都存放在里面