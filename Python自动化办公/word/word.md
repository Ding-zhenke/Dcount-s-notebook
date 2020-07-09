<font face="楷体">

# Python与Word

## create by Dcount

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Python与Word](#python与word)
  - [create by Dcount](#create-by-dcount)
- [一、简介](#一简介)
  - [1. 使用模块Python-docx](#1-使用模块python-docx)
  - [2. word文档结构](#2-word文档结构)
- [二、提取文字](#二提取文字)
  - [1. 读取文档](#1-读取文档)
- [三、写入内容](#三写入内容)
  - [1. 添加标题](#1-添加标题)
  - [2. 添加段落](#2-添加段落)
  - [3. 添加分页](#3-添加分页)
  - [4. 添加图片](#4-添加图片)
  - [5. 添加表格](#5-添加表格)
  - [6. 保存](#6-保存)
- [四、 调整样式](#四-调整样式)
  - [1. 文字样式](#1-文字样式)

<!-- /code_chunk_output -->

# 一、简介

## 1. 使用模块Python-docx

导入时使用`import docx`

## 2. word文档结构

1. Document:文档
2. Paragraph:段落
3. Run:文字块

# 二、提取文字


## 1. 读取文档

```python
from docx import Document
doc = Document('filename.docx')
```

- `.text`读取文字内容
- `doc.paragraphs`:读取文段
- `doc.paragraphs[0].text`:读取段落文字
- `run = doc.paragraphs[1].runs.text`提取文字块RUN

# 三、写入内容

## 1. 添加标题

`doc.add_heading('name',level)`，参数为标题名字和标题等级

## 2. 添加段落

`para1=doc.add_paragraph('内容')`
`para.add_run('文字块').bold = True`：添加字后加粗
`para.add_run('文字块').italic = True`: 斜体

## 3. 添加分页

`doc.add_page_break()`

## 4. 添加图片

`doc.add_picture(图片地址，width=doc.shared.Cm(5),height = doc.shared.Cm(5))`只有给出一个尺寸，其他的可以自动调整

## 5. 添加表格

`doc.add_table(rows,cols)`几行几列的表格
代码示例，假设有数据records

```python
table = doc.add_table(rows=4,cols=3)
for row in range(4):
    cells = table.row[row].cells
    for col in range(3):
        cell[col].text = str(record[row][col])
```

## 6. 保存

`doc.save(文件路径)`

生成一个请假条


# 四、 调整样式

## 1. 文字样式

```python
from docx import Document
from docx.shared import Pt,RGBColor
from docx.oxml.ns import qn
```

`run.font`

