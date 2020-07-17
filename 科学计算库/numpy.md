<font face = "楷体">

---

# 一、数组创建

---

## 1. Ones 和 zeros 填充方式

|方法|描述（N为行，M为列）|
|---|---|
|`empty(shape[, dtype, order])`|返回给定形状和|类型的新数组，而无需初始化条目。|
|`empty_like(prototype[, dtype, order, subok, …]`|返回形状和类型与给定数组相同的新数组。|
|`eye([N, M, k, dtype, order])`|返回一个二维数组，对角线上有一个，其他地方为零。k为0表示主对角线，政治|
|`identity([n, dtype])`|返回标识数组。|
|`ones([shape, dtype, order])`|返回给定形状和类型的新数组，并填充为1。|
|`ones_like([a, dtype, order, subok, shape])`|返回形状与类型与给定数组相同的数组。|
|`zeros([shape, dtype, order])`|返回给定形状和类型的新数组，并用零填充。|
|`zeros_like([a, dtype, order, subok, shape])`|返回形状与类型与给定数组相同的零数组。|
|`full(shape, [fill_value, dtype, order])`|返回给定形状和类型的新数组，并用fill_value填充。|
|`full_like(a, [fill_value, dtype, order, …])`|返回形状和类型与给定数组相同的完整数组。|

## 2. 从现有的数据创建

|方法|描述|
|---|---|
|`array([object, dtype, copy, order, subok, ndmin])`|创建一个数组。|
|`asarray([a, dtype, order])`|将输入转换为数组。
|`asanyarray([a, dtype, order])`|将输入转换为ndarray，但通过ndarray子类。
|`ascontiguousarray([a, dtype])`|返回内存中的连续数组（ndim > = 1）（C顺序）。
|`asmatrix([data, dtype])`|将输入解释为矩阵。
copy(a[, order])`|返回给定对象的数组副本。
|`frombuffer([buffer, dtype, count, offset])`|将缓冲区解释为一维数组。
|`fromfile([file, dtype, count, sep, offset])`|根据文本或二进制文件中的数据构造一个数组。
|`fromfunction(function, shape, **kwargs)`|通过在每个坐标上执行一个函数来构造一个数组。
|`fromiter(iterable, [dtype, count])`|从可迭代对象创建一个新的一维数组。
|`fromstring([string, dtype, count, sep])`|从字符串中的文本数据初始化的新一维数组。
|`loadtxt([fname, dtype, comments, delimiter, …])`|从文本文件加载数据。

## 3. 创建记录数组（numpy.rec）

>注意:numpy.rec 是的首选别名 numpy.core.records。

|方法|	描述|
|--|--|
|`core.records.array([obj, dtype, shape, …])`|从各种各样的对象构造一个记录数组。|
|`core.records.fromarrays(arrayList[, dtype, …])`|从（平面）数组列表创建记录数组|
|`core.records.fromrecords(recList[, dtype, …])`|从文本格式的记录列表创建一个rearray|
|`core.records.fromstring(datastring[, dtype, …])`|根据字符串中包含的二进制数据创建（只读）记录数组|
|`core.records.fromfile(fd[, dtype, shape, …])`|根据二进制文件数据创建数组|

## 4. 创建字符数组（numpy.char）

>注意：`numpy.char`是`numpy.core.defchararray`的首选别名。

|方法|描述|
|--|--|
|`core.defchararray.array(obj[, itemsize, …])`|创建一个chararray。|
|`core.defchararray.asarray(obj[, itemsize, …])`|将输入转换为chararray，仅在必要时复制数据。|

## 5. 数值范围

|方法|描述|
|--|--|
|`arange([start,] stop[, step,][, dtype])`|返回给定间隔内的均匀间隔的值。|
|`linspace(start, stop[, num, endpoint, …])`|返回指定间隔内的等间隔数字。|
|`logspace(start, stop[, num, endpoint, base, …])`|返回数以对数刻度均匀分布。|
|`geomspace(start, stop[, num, endpoint, …])`|返回数字以对数刻度（几何级数）均匀分布。|
|`meshgrid(*xi, **kwargs)`|从坐标向量返回坐标矩阵。|
|`mgrid`|nd_grid实例，它返回一个密集的多维“ meshgrid”。|
|`ogrid`|nd_grid实例，它返回一个开放的多维“ meshgrid”。|

## 6. 创建矩阵

|方法|描述|
|--|--|
|`diag(v[, k])`|提取对角线或构造对角线数组。|
|`diagflat(v[, k])`|使用展平的输入作为对角线创建二维数组。|
|`tri(N[, M, k, dtype])`|在给定对角线处及以下且在其他位置为零的数组。|
|`tril(m[, k])`|数组的下三角。|
|`triu(m[, k])`|数组的上三角。|
|`vander(x[, N, increasing])`|生成范德蒙矩阵|
