import React from 'react';
import { Button, Table, Card, Modal, Divider, Input } from 'antd';
const { Search } = Input;


function showModelDetails(record) {
  console.log(record);
  Modal.info({
    title: record.modelName + "'s Profile",
    width: '50%',
    content: (
      <div style={{ marginTop: 40 }}>
        <div>
          <Divider orientation="left">Serving Information</Divider>
          <div style={{ marginTop: 8 }}>
            <b>Model Name: </b> {record.modelName}
          </div>
          <div style={{ marginTop: 8 }}>
            <b>Model Id: </b>
            {record.modelId}
          </div>
          <div style={{ marginTop: 8 }}>
            <b>Model Framework: </b> {record.modelFramework}
          </div>
          <div style={{ marginTop: 8 }}>
            <b>Serving Engine: </b> {record.modelEngine}
          </div>
          <div style={{ marginTop: 8 }}>
            <b>Serving Device: </b> Nvidia Tesla P4
          </div>
          <br />
        </div>
        <div>
          <Divider orientation="left">Testing Information (finished)</Divider>
          <div style={{ marginTop: 8 }}>
            <b>Testing Batch Size: </b> 64
          </div>
          <div style={{ marginTop: 8 }}>
            <b>Testing Batch Number: </b> 100
          </div>
          <div style={{ marginTop: 8 }}>
            <b>All Batch Throughput: </b> 240.77713166220317 req/sec
          </div>
          <div style={{ marginTop: 8 }}>
            <b>All Batch Latency: </b> 26.580597400665283 sec
          </div>
          <div style={{ marginTop: 8 }}>
            <b>Total GPU Memory: </b> 7981694976.0 bytes
          </div>
          <div style={{ marginTop: 8 }}>
            <b>Average GPU Memory Usage: </b> 0.9726
          </div>
          <div style={{ marginTop: 8 }}>
            <b>Average GPU Memory Used: </b> 7763132416.0 bytes
          </div>
          <div style={{ marginTop: 8 }}>
            <b>Average GPU Utilization: </b> 75.6538%
          </div>
          <br />
        </div>
      </div>
    ),
    onOk() {},
  });
}

const columns = [
  { title: 'Model Id', dataIndex: 'modelId', key: 'modelId' },
  { title: 'Model Name', dataIndex: 'modelName', key: 'modelName' },
  { title: 'Framework', dataIndex: 'modelFramework', key: 'modelFramework' },
  {
    title: 'Pretrained Dataset',
    dataIndex: 'modelDataset',
    key: 'modelDataset',
  },
  { title: 'Accuracy', dataIndex: 'modelAcc', key: 'modelAcc' },
  { title: 'Task', dataIndex: 'modelTask', key: 'modelTask' },
  { title: 'Model User', dataIndex: 'modelUser', key: 'modelUser' },
  {
    title: 'Action',
    dataIndex: '',
    key: 'x',
    render: () => (
      <div>
        <a
          onClick={() => {
            console.log('You clicked edit');
          }}
        >
          Edit
        </a>
        <Divider type="vertical" />
        <a
          onClick={() => {
            console.log('You clicked profile');
          }}
        >
          Profile
        </a>
      </div>
    ),
  },
];

const data = [
  {
    key: 1,
    modelId: '734d623fd24',
    modelName: 'ResNet-50',
    modelFramework: 'PyTorch',
    modelAcc: '84%',
    modelEngine: 'TensorFlow-Serving',
    modelTask: 'image classification',
    modelUser: 'Yizheng Huang',
    modelDataset: 'ImageNet',
    hasDetail: false,
    servingDevice: 'Nvidia Tesla P4',
    allThroughput: '270.341 req/sec',
    allLatency: '26.580597400665283 sec',
    batchSize: 64,
    batchNum: 100,
    totalMem: '7981694976.0 bytes',
    usedMem: '7763132416.0 bytes',
    memPer: '0.9726',
    gpuUtil: '75.6538%',
    latency50: '0.2650185823440552 s',
    latency95: '0.27614104747772217 s',
    latency99: '0.2790443968772889 s',
  },
  {
    key: 2,
    modelId: '7e2562srd24',
    modelName: 'BERT-Medium',
    modelFramework: 'TensorFlow',
    modelAcc: '96%',
    modelEngine: 'ONNX Runtime',
    modelTask: 'text classification',
    modelUser: 'Yuanming Lee',
    modelDataset: 'GLUE',
    hasDetail: true,
    servingDevice: 'Nvidia Tesla P4',
    allThroughput: '270.341 req/sec',
    allLatency: '26.580597400665283 sec',
    batchSize: 64,
    batchNum: 100,
    totalMem: '7981694976.0 bytes',
    usedMem: '7763132416.0 bytes',
    memPer: '0.9726',
    gpuUtil: '75.6538%',
    latency50: '0.2650185823440552 s',
    latency95: '0.27614104747772217 s',
    latency99: '0.2790443968772889 s',
  },
  {
    key: 3,
    modelId: '234d623fsd4',
    modelName: 'ResNet-101',
    modelFramework: 'ONNX',
    modelAcc: '74%',
    modelEngine: 'TensorFlow-Serving',
    modelTask: 'image classification',
    modelUser: 'Huaizheng Zhang',
    modelDataset: 'ImageNet',
    hasDetail: true,
    servingDevice: 'Nvidia Tesla P4',
    allThroughput: '270.341 req/sec',
    allLatency: '26.580597400665283 sec',
    batchSize: 64,
    batchNum: 100,
    totalMem: '7981694976.0 bytes',
    usedMem: '7763132416.0 bytes',
    memPer: '0.9726',
    gpuUtil: '75.6538%',
    latency50: '0.2650185823440552 s',
    latency95: '0.27614104747772217 s',
    latency99: '0.2790443968772889 s',
  },
  {
    key: 4,
    modelId: 'b45d623fsq0',
    modelName: 'ResNet-50',
    modelFramework: 'ONNX',
    modelAcc: '83%',
    modelEngine: 'TensorRT Serving',
    modelTask: 'image classification',
    modelUser: 'Yonggang Wen',
    modelDataset: 'CoCo',
    hasDetail: true,
    servingDevice: 'Nvidia Tesla P4',
    allThroughput: '270.341 req/sec',
    allLatency: '26.580597400665283 sec',
    batchSize: 64,
    batchNum: 100,
    totalMem: '7981694976.0 bytes',
    usedMem: '7763132416.0 bytes',
    memPer: '0.9726',
    gpuUtil: '75.6538%',
    latency50: '0.2650185823440552 s',
    latency95: '0.27614104747772217 s',
    latency99: '0.2790443968772889 s',
  },
  {
    key: 5,
    modelId: 'bs0d453fs2w',
    modelName: 'BERT-Tiny',
    modelFramework: 'PyTorch',
    modelAcc: '53%',
    modelEngine: 'TensorRT Serving',
    modelTask: 'text classification',
    modelUser: 'Yonggang Wen',
    modelDataset: 'GLUE',
    hasDetail: false,
    servingDevice: 'Nvidia Tesla P4',
    allThroughput: '270.341 req/sec',
    allLatency: '26.580597400665283 sec',
    batchSize: 64,
    batchNum: 100,
    totalMem: '7981694976.0 bytes',
    usedMem: '7763132416.0 bytes',
    memPer: '0.9726',
    gpuUtil: '75.6538%',
    latency50: '0.2650185823440552 s',
    latency95: '0.27614104747772217 s',
    latency99: '0.2790443968772889 s',
  },
  {
    key: 6,
    modelId: '1f3d453fs3l',
    modelName: 'ResNet-101',
    modelFramework: 'TensorRT',
    modelAcc: '67%',
    modelEngine: 'TorchScript',
    modelTask: 'image classification',
    modelUser: 'Huaizheng Zhang',
    modelDataset: 'Coco',
    hasDetail: true,
    servingDevice: 'Nvidia Tesla P4',
    allThroughput: '270.341 req/sec',
    allLatency: '26.580597400665283 sec',
    batchSize: 64,
    batchNum: 100,
    totalMem: '7981694976.0 bytes',
    usedMem: '7763132416.0 bytes',
    memPer: '0.9726',
    gpuUtil: '75.6538%',
    latency50: '0.2650185823440552 s',
    latency95: '0.27614104747772217 s',
    latency99: '0.2790443968772889 s',
  },
];

const Dashboard = () => {
  return (
    <Card>
      <div
        style={{ marginBottom: '20px', display: 'flex', flexDirection: 'row' }}
      >
        <Button
          type="primary"
          onClick={() => {
            console.log('You clicked Register Model');
          }}
        >
          Register Model
        </Button>
        <Button
          style={{ marginLeft: '5px' }}
          onClick={() => {
            console.log('You clicked Download Table');
          }}
        >
          Download Table
        </Button>
        <Search
          style={{ marginLeft: '10px' }}
          placeholder="search model record by key words"
          enterButton="Search"
          size="samll"
          onSearch={(value) => console.log(value)}
        />
      </div>
      <Divider dashed />
      <Table
        columns={columns}
        dataSource={data}
        expandable={{
          expandedRowRender: (record) => (
            <div>
              <p style={{ margin: '5px' }}>Model Name: {record.modelName}</p>
              <p style={{ margin: '5px' }}>
                Pretrained Dataset: {record.modelDataset}
              </p>
              <p style={{ margin: '5px' }}>
                Serving Engine: {record.modelEngine}
              </p>
              <p style={{ margin: '5px' }}>
                Serving Device: {record.servingDevice}
              </p>
              <p style={{ margin: '5px' }}>
                Tested Batch Size: {record.batchSize}
              </p>
              <p style={{ margin: '5px' }}>
                Tested Batch Number: {record.batchNum}
              </p>
              <p style={{ margin: '5px' }}>
                Tested Batch Number: {record.batchNum}
              </p>
              <p style={{ margin: '5px' }}>
                Total Memory of Device: {record.totalMem}
              </p>
              <p style={{ margin: '5px' }}>
                Total Memory Used: {record.usedMem} 
              </p>
              <p style={{ margin: '5px' }}>
                Total Memory Loaded Percentile: {record.memPer} 
              </p>
              <p style={{ margin: '5px' }}>
                Total GPU Utilization: {record.gpuUtil} 
              </p>
              <p style={{ margin: '5px' }}>
                All Batch Latency: {record.allLatency} 
              </p>
              <p style={{ margin: '5px' }}>
                Overall 50th Percentile Latency: {record.latency50} 
              </p>
              <p style={{ margin: '5px' }}>
                Overall 95th Percentile Latency: {record.latency95} 
              </p>
              <p style={{ margin: '5px' }}>
                Overall 99th Percentile Latency: {record.allThroughput} 
              </p>
              <p style={{ margin: '5px' }}>
                All Batch Throughput: {record.allThroughput} 
              </p>
            </div>
          ),
          rowExpandable: (record) => record.hasDetail,
        }}
        // onRow={(record) => ({
        //   onClick: () => {
        //     showModelDetails(record);
        //   },
        // })}
      />
    </Card>
  );
};

export default Dashboard;
