import React from 'react';
import {
  Button,
  Table,
  Card,
  Modal,
  Divider,
  Input,
  Radio,
  Descriptions,
  Tag,
} from 'antd';

import './index.css';

const { Search } = Input;

function showModelDetails(record) {
  console.log(record);
  Modal.info({
    title: `${record.modelName}'s Profile`,
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
  // { title: 'Model Id', dataIndex: 'modelId', key: 'modelId' },
  {
    title: 'Model Name',
    dataIndex: 'modelName',
    key: 'modelName',
    className: 'column',
  },
  {
    title: 'Framework',
    dataIndex: 'modelFramework',
    key: 'modelFramework',
    className: 'column',
  },
  {
    title: 'Pretrained Dataset',
    dataIndex: 'modelDataset',
    key: 'modelDataset',
    className: 'column',
  },
  {
    title: 'Accuracy',
    dataIndex: 'modelAcc',
    key: 'modelAcc',
    className: 'column',
  },
  {
    title: 'Task',
    dataIndex: 'modelTask',
    key: 'modelTask',
    className: 'column',
  },
  {
    title: 'Model User',
    dataIndex: 'modelUser',
    key: 'modelUser',
    className: 'column',
  },
  {
    title: 'Action',
    dataIndex: '',
    key: 'x',
    className: 'column',
    render: () => (
      <div>
        {/* <Radio.Group size="large">
          <Radio.Button
            value="large"
            onClick={() => {
              console.log('You clicked edit');
            }}
          >
            Edit
          </Radio.Button>
          <Radio.Button
            value="large"
            onClick={() => {
              console.log('You clicked profile');
            }}
          >
            Profile
          </Radio.Button>
        </Radio.Group> */}
        <Button type="primary" size="large">
          Edit
        </Button>
        <Button style={{ marginLeft: '3px' }} type="primary" size="large">
          Profile
        </Button>
      </div>
    ),
  },
];

const data = [
  {
    key: 1,
    modelId: '734d623fd24',
    modelName: 'ResNet-50',
    modelFramework: 'TensorFlow',
    convertVersion: 'TensorFlow-SavedModel',
    modelAcc: '75.3%',
    modelEngine: 'TensorFlow-Serving',
    modelTask: 'image classification',
    modelUser: 'Yizheng Huang',
    modelDataset: 'ImageNet',
    hasDetail: true,
    servingDevice: 'Nvidia Tesla P4',
    allThroughput: '200.333 req/sec',
    allLatency: '31.946 sec',
    batchSize: 16,
    batchNum: 100,
    totalMem: '7981694976.0 bytes',
    usedMem: '7763132416.0 bytes',
    memPer: '97.263%',
    gpuUtil: '69.871%',
    latency50: '78.906 ms',
    latency95: '84.170 ms',
    latency99: '86.797 ms',
  },
  {
    key: 2,
    modelId: '7e2562srd24',
    modelName: 'BERT-Medium',
    modelFramework: 'PyTorch',
    convertVersion: 'TensorFlow-SavedModel',
    modelAcc: '71.0%',
    modelEngine: 'ONNX Runtime',
    modelTask: 'text classification',
    modelUser: 'Yuanming Lee',
    modelDataset: 'GLUE',
    hasDetail: false,
    servingDevice: 'Nvidia Tesla P4',
    allThroughput: '270.341 req/sec',
    allLatency: '26.580 sec',
    batchSize: 64,
    batchNum: 100,
    totalMem: '7981694976.0 bytes',
    usedMem: '7763132416.0 bytes',
    memPer: '0.972',
    gpuUtil: '75.653%',
    latency50: '0.265 s',
    latency95: '0.276 s',
    latency99: '0.279 s',
  },
  {
    key: 1,
    modelId: '734d623fd24',
    modelName: 'ResNet-50',
    modelFramework: 'PyTorch',
    convertVersion: 'TensorFlow-SavedModel',
    modelAcc: '70.1%',
    modelEngine: 'TorchScript',
    modelTask: 'image classification',
    modelUser: 'Yizheng Huang',
    modelDataset: 'ImageNet',
    hasDetail: false,
    servingDevice: 'Nvidia Tesla P4',
    allThroughput: '200.333 req/sec',
    allLatency: '31.946 sec',
    batchSize: 16,
    batchNum: 100,
    totalMem: '7981694976.0 bytes',
    usedMem: '7763132416.0 bytes',
    memPer: '97.263%',
    gpuUtil: '69.871%',
    latency50: '78.906 ms',
    latency95: '84.170 ms',
    latency99: '86.797 ms',
  },
];

const Dashboard = () => {
  return (
    <Card>
      <div
        style={{ marginBottom: '20px', display: 'flex', flexDirection: 'row' }}
      >
        <Button
          size="large"
          type="primary"
          onClick={() => {
            console.log('You clicked Register Model');
          }}
        >
          Register Model
        </Button>
        <Button
          size="large"
          style={{ marginLeft: '5px' }}
          onClick={() => {
            console.log('You clicked Download Table');
          }}
        >
          Download Table
        </Button>
        <Search
          size="large"
          style={{ marginLeft: '10px' }}
          placeholder="search model record by key words"
          enterButton="Search"
          onSearch={(value) => console.log(value)}
        />
      </div>
      <Divider dashed />
      <Table
        columns={columns}
        dataSource={data}
        expandable={{
          expandedRowRender: (record) => (
            <div style={{ backgroundColor: '#F5F5F5', padding: '10px' }}>
              <Descriptions
                style={{ width: '92%', margin: '0 auto' }}
                column={3}
                size="middle"
                title={
                  <a
                    style={{
                      whiteSpace: 'nowrap',
                      fontSize: 25,
                      color: 'black',
                    }}
                  >
                    Converted Model Info
                  </a>
                }
              >
                <Descriptions.Item
                  label={
                    <a
                      style={{
                        whiteSpace: 'nowrap',
                        fontSize: 25,
                        color: 'black',
                      }}
                    >
                      Model Name
                    </a>
                  }
                >
                  <Tag
                    color="volcano"
                    style={{
                      height: '25px',
                      textAlign: 'center',
                      fontSize: 25,
                    }}
                  >
                    {record.modelName}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item
                  label={
                    <a
                      style={{
                        whiteSpace: 'nowrap',
                        fontSize: 25,
                        color: 'black',
                      }}
                    >
                      Converted Version
                    </a>
                  }
                >
                  <Tag
                    color="blue"
                    style={{
                      height: '25px',
                      textAlign: 'center',
                      fontSize: 25,
                    }}
                  >
                    {record.convertVersion}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item
                  label={
                    <a
                      style={{
                        whiteSpace: 'nowrap',
                        fontSize: 25,
                        color: 'black',
                      }}
                    >
                      Serving Engine
                    </a>
                  }
                >
                  <Tag
                    color="pink"
                    style={{
                      height: '25px',
                      textAlign: 'center',
                      fontSize: 25,
                    }}
                  >
                    {record.modelEngine}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item
                  span={2}
                  label={
                    <a
                      style={{
                        whiteSpace: 'nowrap',
                        fontSize: 25,
                        color: 'black',
                      }}
                    >
                      Conversion State
                    </a>
                  }
                >
                  <a
                    style={{
                      fontSize: 25,
                      whiteSpace: 'nowrap',
                      color: 'green',
                    }}
                  >
                    Success
                  </a>
                  <Divider type="vertical" />
                  {/* <Radio.Group size="large">
                    <Radio.Button
                      value="large"
                      onClick={() => {
                        console.log('You clicked Deploy');
                      }}
                    >
                      Deploy this Model
                    </Radio.Button>
                  </Radio.Group> */}

                  <Button type="primary" size="large">
                    Deploy this Model
                  </Button>
                </Descriptions.Item>
              </Descriptions>
              <Descriptions
                style={{ width: '92%', margin: '0 auto' }}
                column={3}
                size="small"
                title={
                  <a
                    style={{
                      whiteSpace: 'nowrap',
                      fontSize: 25,
                      color: 'black',
                    }}
                  >
                    Profiling Results
                  </a>
                }
              >
                <Descriptions.Item
                  label={
                    <a
                      style={{
                        whiteSpace: 'nowrap',
                        fontSize: 25,
                        color: 'black',
                      }}
                    >
                      Profiling Device
                    </a>
                  }
                >
                  <Tag
                    color="geekblue"
                    style={{
                      height: '25px',
                      textAlign: 'center',
                      fontSize: 25,
                    }}
                  >
                    {record.servingDevice}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item
                  label={
                    <a
                      style={{
                        whiteSpace: 'nowrap',
                        fontSize: 25,
                        color: 'black',
                      }}
                    >
                      Profiling Batch Size
                    </a>
                  }
                >
                  <Tag
                    color="gold"
                    style={{
                      height: '25px',
                      textAlign: 'center',
                      fontSize: 25,
                    }}
                  >
                    {record.batchSize}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item
                  label={
                    <a
                      style={{
                        whiteSpace: 'nowrap',
                        fontSize: 25,
                        color: 'black',
                      }}
                    >
                      Profiling Number of Batches
                    </a>
                  }
                >
                  <Tag
                    color="blue"
                    style={{
                      height: '25px',
                      textAlign: 'center',
                      fontSize: 25,
                    }}
                  >
                    {record.batchNum}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item
                  label={
                    <a
                      style={{
                        whiteSpace: 'nowrap',
                        fontSize: 25,
                        color: 'black',
                      }}
                    >
                      All Batch Latency
                    </a>
                  }
                >
                  <Tag
                    color="cyan"
                    style={{
                      height: '25px',
                      textAlign: 'center',
                      fontSize: 25,
                    }}
                  >
                    {record.allLatency}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item
                  span={2}
                  label={
                    <a
                      style={{
                        whiteSpace: 'nowrap',
                        fontSize: 25,
                        color: 'black',
                      }}
                    >
                      All Batch Throughput
                    </a>
                  }
                >
                  <Tag
                    color="geekblue"
                    style={{
                      height: '25px',
                      textAlign: 'center',
                      fontSize: 25,
                    }}
                  >
                    {record.allThroughput}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item
                  label={
                    <a
                      style={{
                        whiteSpace: 'nowrap',
                        fontSize: 25,
                        color: 'black',
                      }}
                    >
                      P50 Latency
                    </a>
                  }
                >
                  <Tag
                    color="gold"
                    style={{
                      height: '25px',
                      textAlign: 'center',
                      fontSize: 25,
                    }}
                  >
                    {record.latency50}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item
                  label={
                    <a
                      style={{
                        whiteSpace: 'nowrap',
                        fontSize: 25,
                        color: 'black',
                      }}
                    >
                      P95 Latency
                    </a>
                  }
                >
                  <Tag
                    color="green"
                    style={{
                      height: '25px',
                      textAlign: 'center',
                      fontSize: 25,
                    }}
                  >
                    {record.latency95}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item
                  label={
                    <a
                      style={{
                        whiteSpace: 'nowrap',
                        fontSize: 25,
                        color: 'black',
                      }}
                    >
                      P99 Latency
                    </a>
                  }
                >
                  <Tag
                    color="pink"
                    style={{
                      height: '25px',
                      textAlign: 'center',
                      fontSize: 25,
                    }}
                  >
                    {record.latency99}
                  </Tag>
                </Descriptions.Item>
              </Descriptions>
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
