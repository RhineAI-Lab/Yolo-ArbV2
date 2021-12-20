<div align="center">
<p>
   <img width="850" src="https://github.com/HRan2004/Yolo-ArbV2/data/images/ResultShow.jpg"></a>
</p>
<p>
Yolo-ArbV2 在 <a href="https://github.com/ultralytics/yolov5">YOLOv5</a> 基础上进行二次开发。<br/>
保持GT框检测功能的同时，新增了额外输出信息，用于检测输出中目标多边形的信息。这样实现了基于矩形Anchor-based多边形检测功能。
</p>
</div>

## <div align="center">文档</div>

大部分操作方法可参考 [YOLOv5 文档](https://docs.ultralytics.com) 。新增区别信息在下文会介绍。

## <div align="center">快速开始</div>

<details open>
<summary>Install</summary>

[**Python>=3.6.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->

```bash
$ git clone https://github.com/HRan2004/yolov5
$ cd yolov5
$ pip install -r requirements.txt
```

</details>

<details>
<summary>Datasets</summary>
数据集的准备，大体结构与YOLOv5一致。目录结构如下。<br/>

```text
datasets
├ images
│ └ img1,img2...
├ labels
│ └ txt1,txt2...
├ train.txt
└ val.txt 
```

images与labels文件夹中存放图片与标签，train.txt与val.txt用于分类图片作为训练集或测试集。存放图片路径，一行一个，会自动计算标签路径。<br/>

<br/>

区别在于txt文件中的坐标信息。 当输出边数，即参数edges为4时，用于检测4边形，数据集需要四点坐标。<br/>
注意：坐标宽高，全都为相对图片宽高归一化后的数据，即0到1范围内
```text
YOLOv5: class center_x center_y width height

Yolo-ArbV2: class x1 y1 x2 y2 x3 y3 x4 y4
```

</details>

<details>
<summary>Args</summary>

在hyp中新增了5项参数。
```yaml
edges: 4 # edges of poly for net output
poly: 1.2  # obj loss gain (scale with pixels)
start_poly: 0 # epoch to start use poly loss
poly_out: 0.5 # max percent for poly out of box
poly_loss_smooth: 0.005 # smooth scope for SmoothL1Loss
```
edges 多边形边数，必填。用于设置模型输出多边形信息的边数，识别四边形则为4，需与数据集一致。设置为0时，模型功能等同于YOLOv5。<br/>
poly 多边形框损失。多边形损失权重。<br/>
start_poly 开始poly损失的epoch。由于poly相对于box进行归一化输出，建议box准确后再进行训练poly。<br/>
poly_out 出框量。poly可溢出box的最大比值。由于数据增强对box框的切割，实际poly位置可能溢出box。<br/>
poly_loss_smooth 损失平滑范围，后期将删除该参数。由于输出结果归一化，使用SmoothL1Loss需减少平滑区范围。建议填写0.005。
</details>



<details>
<summary>Args</summary>

在hyp中新增了5项参数。
```yaml
edges: 4 # edges of poly for net output
poly: 1.2  # obj loss gain (scale with pixels)
start_poly: 0 # epoch to start use poly loss
poly_out: 0.5 # max percent for poly out of box
poly_loss_smooth: 0.005 # smooth scope for SmoothL1Loss
```
edges 多边形边数，必填。用于设置模型输出多边形信息的边数，识别四边形则为4，需与数据集一致。设置为0时，模型功能等同于YOLOv5。<br/>
poly 多边形框损失。多边形损失权重。<br/>
start_poly 开始poly损失的epoch。由于poly相对于box进行归一化输出，建议box准确后再进行训练poly。<br/>
poly_out 出框量。poly可溢出box的最大比值。由于数据增强对box框的切割，实际poly位置可能溢出box。<br/>
poly_loss_smooth 损失平滑范围，后期将删除该参数。由于输出结果归一化，使用SmoothL1Loss需减少平滑区范围。建议填写0.005。
</details>

## <div align="center">理论介绍</div>

<details>
<summary>Model</summary>

模型结构维持不变，仅改变模型输出层维度。后期可能适当增加输出层处理层数。<br/>
```text
输出层 每个框：
YOLOv5: x,y,w,h,conf,class1,...,classF
Yolo-ArbV2(edges=N): x,y,w,h,[x1,y1,...,xN,yN],conf,class1,...,classF
```
每框信息量为: edges*2 + class_num + 5

</details>

<details>
<summary>Processing</summary>

对于多边形信息的后处理。poly信息相对于box位置进行归一化输出。<br/>
相对于box左上角点，故输出区间为 [-poly_out,1+poly_out]。<br/>
输出结果，进行Sigmoid处理后，*（1+poly_out*2）- poly_out即可得到输出结果。

</details>

<details>
<summary>Loss</summary>

损失的处理由于多边形的IOU计算复杂，效率过低。模型设计了多点距离法计算多边形损失。直接计算对应位置输出坐标与真实坐标的距离，将8个参数一同进行SmoothL1Loss损失，得出结果。<br/>
为避免点位顺序不同导致的损失异常，模型在导入数据数据增强后（可能包含旋转影响顺序），会自动将数据集处理成，从最高点开始顺时针点位排序的数据集。<br/>
<br/>
新增了SmoothL1LossSr，可自定义损失的平滑范围，提高精确度。
```python
class SmoothL1LossSr(nn.SmoothL1Loss):
    def __init__(self,smooth_range=1.0):
        super().__init__()
        self.sr = smooth_range

    def __call__(self, p, t):
        sr = self.sr
        loss = super().__call__(p/sr, t/sr)*sr
        return loss
```

</details>


<details>
<summary>Datasets mosaic</summary>
数据增强过程中，跟随box框，共同对poly进行处理。并在最后根据poly生成准确box框（旋转变换会导致box不准确）。<br/>
<br/>
值得一提的是，修复了一项原有Yolov5中对segment生成box框时的bug。<br/>
在切割变换中，Yolov5通过在多边形上描绘2000个点，根据剩余点生成box。但是过程中遗漏了终点和起点连接的边线，这导致再切割相邻边线时，无法准确生成box框。<br/>
<img width="850" src="https://github.com/HRan2004/Yolo-ArbV2/data/images/debug01.jpg"></a>
如上图，缺少上边线，并且右边线遭到切割时，生成的box出现了错误。<br/>
<br/>
解决方法也很简单，在描边前进行补充起点在终点后方即可。

```python
def resample_segments(segments, n=500):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n) # Debug
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments
```

</details>


<details>
<summary>Datasets & Loss</summary>
为合理计算损失，在数据增强时，会有旋转翻折等情况，数据点位顺序可能错乱，所以在数据增强完毕后，要进行顺序处理。<br/>
最终转换为由最高点为起点，顺时针旋转的多边形数据。这使得直接计算对应位置点位距离的损失方法可行了。<br/>
以下为比较简洁的批量实现方式

```python
# Make polygons start with the highest point
# and order with clock wise
# Input shape : [polygons_num,edges,2(x,y)]
# Output shape : ==Input shape
def polygons_cw(polys):
    ps = polys.shape
    hpi = np.argmax(polys[..., 1::2], axis=1).reshape(polys.shape[0]) # Highest point index
    lpi,rpi = hpi-1,hpi+1
    np.place(lpi,lpi==-1,ps[1]-1)
    np.place(rpi,rpi==ps[1],0)
    lrpi = np.concatenate(([lpi], [rpi]),0).T.flatten() # Left right point by highest point
    p1 = polys.reshape((-1,2))[hpi+np.arange(ps[0])*ps[1]].repeat(2,axis=0).reshape(-1,2,2)
    p2 = polys.reshape((-1,2))[lrpi+np.arange(ps[0]).repeat(2)*ps[1]].reshape((-1,2,2))
    pc = p2-p1
    d = 90-np.arctan(pc[...,0]/pc[...,1])*180/math.pi
    isCw = d[...,0]<d[...,1] # Is clock wise
    polys_cw = np.zeros_like(polys)
    for i in range(polys.shape[0]):
        hpii = hpi[i]
        if isCw[i]:
            polys_cw[i,:ps[1]-hpii] = polys[i,hpii:]
            polys_cw[i,ps[1]-hpii:] = polys[i,:hpii]
        else:
            polys_cw[i,:hpii+1] = polys[i,hpii::-1]
            polys_cw[i,hpii+1:] = polys[i,:hpii-ps[1]:-1]
    return polys_cw
```

</details>
