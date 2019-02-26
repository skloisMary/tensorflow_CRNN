# tensorflow_CRNN
CNN+RNN+CTC

论文：An End-to-ENd Trainable Neural Network for Image-based Sequence Recognition and Its application to Scene Text Recognition 的tensorflow实现

<table>
    <thead>
        <tr>
            <th>文件</th>
            <th>作用</th>
        </tr>
    </thead>
    <tbody>
       <tr> <th>extract.py</th>  <th>提取图片标签中的字符，并制成map.json文件</th> </tr>
       <tr> <th>TFrecorde.py </th>  <th> 把数据集生成tfrecorde文件</th> </tr>
       <tr> <th>Model.py </th>  <th> 定义模型，train/test</th> </tr>
       <tr> <th>run.py </th>  <th>主程序</th> </tr>
       <tr> <th>inference.py </th>  <th> 使用新数据，引用模型</th> </tr>
    </tbody>
</table>
