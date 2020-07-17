import os 

`os.getcwd()` 当前文件路径
`os.name` 当前系统
>window-nt
>linux-posix


`os.mkdir('路径')`创建文件夹

`./`当前文件路劲

`os.path.abspath('路劲')`返回绝对路径

`os.path`

`os.path.isdir('./')`判断是否是文件夹
`os.path.isdfile('./')`判断是否是文件

`os.path.splitext('filename')`拆分文件名和后缀名，路径也会和文件名贴一起；需要保证文件存在；

`os.path.exists('filename')`判断是否存在

`os.path.getctime(filename)`获取创建时间,返回来的是时间戳。

时间戳变成时间

```python
import time 
t=time.localtime("时间戳")
myt=time.strftime('%Y-%m-%d %H:%M-%S',t)
print(myt)
```

`os.listdir(路径)`返回路径下所有文件夹和文件名，隐藏文件可以看到

`os.path.join(拼接文件1,2)`一般1是文件夹，2是文件可以有多个

`os.rename(old,new)` 改名字

