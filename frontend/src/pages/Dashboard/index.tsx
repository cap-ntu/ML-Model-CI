import React from 'react';
import { Button, Table, Card, Divider, Input, Descriptions, Tag } from 'antd';
import axios from 'axios';
import reqwest from 'reqwest';
import './index.css';

const { Search } = Input;
const columns = [
  {
    title: 'Model Name',
    dataIndex: 'name',
    key: 'name',
    className: 'column',
  },
  {
    title: 'Framework',
    dataIndex: 'framework',
    key: 'framework',
    className: 'column',
  },
  {
    title: 'Pre-trained Dataset',
    dataIndex: 'dataset',
    key: 'dataset',
    className: 'column',
  },
  {
    title: 'Accuracy',
    dataIndex: 'acc',
    key: 'acc',
    className: 'column',
  },
  {
    title: 'Task',
    dataIndex: 'task',
    key: 'task',
    className: 'column',
  },
  // {
  //   title: 'Model User',
  //   dataIndex: 'modelUser',
  //   key: 'modelUser',
  //   className: 'column',
  // },
  {
    title: 'Status',
    dataIndex: 'status',
    key: 'status',
    className: 'column',
  },
  {
    title: 'Action',
    dataIndex: '',
    key: 'x',
    className: 'column',
    render: () => (
      <div>
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

const getRandomuserParams = (params) => {
  return {
    results: params.pagination.pageSize,
    page: params.pagination.current,
    ...params,
  };
};

export default class Dashboard extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      allModelInfo: [],
      pagination: {
        current: 1,
        pageSize: 10,
      },
      loading: false,
    };
    this.loadAllModels();
  }

  componentDidMount() {
    const { pagination } = this.state;
    this.fetch({ pagination });
  }

  handleTableChange = (pagination, filters, sorter) => {
    this.fetch({
      sortField: sorter.field,
      sortOrder: sorter.order,
      pagination,
      ...filters,
    });
  };

  fetch = (params = {}) => {
    this.setState({ loading: true });
    reqwest({
      url: 'http://155.69.146.35:8000/api/v1/model/',
      method: 'get',
      type: 'json',
      data: getRandomuserParams(params),
    }).then((allModelInfo) => {
      this.setState({
        loading: false,
        data: allModelInfo.data,
        pagination: {
          ...params.pagination,
          total: this.state.allModelInfo.length,
        },
      });
    });
  };

  loadAllModels = () => {
    const targetUrl = 'http://155.69.146.35:8000/api/v1/model/';
    axios
      .get(targetUrl)
      .then((response) => {
        // handle success
        // console.log(response.data);
        this.setState({ allModelInfo: response.data });
      })
      .catch((error) => {
        // handle error
        // console.log(error);
      })
      .then(() => {
        // always executed
      });
  };

  loadModelById = (id) => {
    const targetUrl = 'http://155.69.146.35:8000/api/v1/model/';
    axios
      .get(targetUrl + id)
      .then((response) => {
        // handle success
        return response;
      })
      .catch((error) => {
        // handle error
        // console.log(error);
      })
      .then(() => {
        // always executed
      });
    return [];
  };

  public render() {
    return (
      <Card>
        <div
          style={{
            marginBottom: '20px',
            display: 'flex',
            flexDirection: 'row',
          }}
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
          rowKey={(record) => record.id}
          columns={columns}
          pagination={this.state.pagination}
          loading={this.state.loading}
          onChange={this.handleTableChange}
          dataSource={this.state.allModelInfo}
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
                      {record.name}
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
                      {record.framework}
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
                      {record.engine}
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
                      {record.status}
                    </a>
                    <Divider type="vertical" />
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
            rowExpandable: (record) => true,
          }}
        />
      </Card>
    );
  }
}
