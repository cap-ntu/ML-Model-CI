import React from 'react';
import {
  Button,
  Table,
  Card,
  Divider,
  Input as search,
  Descriptions,
  Tag,
  Menu,
  Dropdown,
  Modal,
  Tooltip
} from 'antd';
import {
  EditOutlined,
  ProfileOutlined,
  BranchesOutlined,
} from '@ant-design/icons';
import { SchemaForm, FormButtonGroup, Submit, Reset } from '@formily/antd';
import axios from 'axios';
import { config, Link } from 'ice';
import reqwest from 'reqwest';
import './index.css';
import { Input, Form } from 'antd';
import {
  Select,
  Switch,
  NumberPicker,
  FormMegaLayout,
  DatePicker,
  FormLayout,
  ArrayTable,
} from '@formily/antd-components';
// eslint-disable-next-line @typescript-eslint/no-var-requires
const registerSchema = require('./utils/schema.json');

const components = {
  Input,
  Select,
  Switch,
  NumberPicker,
  FormMegaLayout,
  DatePicker,
  FormLayout,
  ArrayTable,
};

const { Search } = search;

const tagColor = {
  Published: 'geekblue',
  Converted: 'cyan',
  Profiling: 'purple',
  'In Service': 'lime',
  Draft: 'red',
  Validating: 'magenta',
  Training: 'volcano',
};

const columns = [
  {
    title: 'Architecture',
    dataIndex: 'architecture',
    key: 'architecture',
    className: 'column',
    ellipsis: {
      showTitle: false,
    },
    render: architecture => (
      <Tooltip placement="topLeft" title={architecture}>
        {architecture}
      </Tooltip>
    )
  },
  {
    title: 'Framework',
    dataIndex: 'framework',
    key: 'framework',
    className: 'column',
    ellipsis: {
      showTitle: false,
    },
    render: framework => (
      <Tooltip placement="topLeft" title={framework}>
        {framework}
      </Tooltip>
    )
  },
  {
    title: 'Engine',
    dataIndex: 'engine',
    key: 'engine',
    className: 'column',
    ellipsis: {
      showTitle: false,
    },
    render: engine => (
      <Tooltip placement="topLeft" title={engine}>
        {engine}
      </Tooltip>
    )
  },
  {
    title: 'Pre-trained Dataset',
    dataIndex: 'dataset',
    key: 'dataset',
    ellipsis: {
      showTitle: false,
    },
    className: 'column',
    render: dataset => (
      <Tooltip placement="topLeft" title={dataset}>
        {dataset}
      </Tooltip>
    ),
  },
  {
    title: 'Metric',
    dataIndex: 'metric',
    key: 'metric',
    className: 'column',
    ellipsis: {
      showTitle: false,
    },
    render: metric => (
      <Tooltip placement="topLeft" title={Object.keys(metric)[0]}>
        {Object.keys(metric)[0]}
      </Tooltip>
    )
  },
  {
    title: 'Score',
    dataIndex: 'metric',
    key: 'score',
    className: 'column',
    ellipsis: {
      showTitle: false,
    },
    render: metric => (
      <Tooltip placement="topLeft" title={metric[Object.keys(metric)[0]]}>
        {metric[Object.keys(metric)[0]]}
      </Tooltip>
    )
  },
  {
    title: 'Task',
    dataIndex: 'task',
    key: 'task',
    className: 'column',
    ellipsis: {
      showTitle: false,
    },
    render: task => (
      <Tooltip placement="topLeft" title={task}>
        {task}
      </Tooltip>
    )
  },
  {
    title: 'Version',
    dataIndex: 'version',
    key: 'version',
    className: 'column',
    ellipsis: {
      showTitle: false,
    },
    render: version => (
      <Tooltip placement="topLeft" title={version}>
        {version}
      </Tooltip>
    )
  },
  {
    title: 'Status',
    dataIndex: 'model_status',
    key: 'model_status',
    className: 'column',
    ellipsis: {
      showTitle: false,
    },
    render: (status) => (
      <Tooltip placement="topLeft" title={status}>
        <Tag color={tagColor[status]}>
          {status}
        </Tag>
      </Tooltip>
    )
  },
  {
    title: 'Model User',
    dataIndex: 'creator',
    key: 'creator',
    className: 'column',
    ellipsis: {
      showTitle: false,
    },
    render: creator => (
      <Tooltip placement="topLeft" title={creator}>
        {creator}
      </Tooltip>
    )
  },
  {
    title: 'Action',
    dataIndex: 'id',
    key: 'id',
    className: 'column',
    fixed: 'right',
    responsive: ["sm"],
    render: (text, record) => {
      const menu = (
        <Menu>
          <Menu.Item
            key="2"
            icon={<ProfileOutlined />}
            style={{ fontSize: 18 }}
          >
            Profile
          </Menu.Item>
          {record.engine === 'PYTORCH' || record.engine === 'TFS' ? (
            <Menu.Item
              key="3"
              icon={<BranchesOutlined />}
              style={{ fontSize: 18 }}
            >
              <Link to={`/visualizer/${text}`}>Finetune</Link>
            </Menu.Item>
          ) : (
            ''
          )}
        </Menu>
      );
      return (
        <Dropdown.Button overlay={menu} size="large">
          <EditOutlined /> Edit
        </Dropdown.Button>
      );
    },
  }
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
      profilingResult: {},
      allModelInfo: [],
      pagination: {
        current: 1,
        pageSize: 10,
      },
      loading: false,
      showRegisterForm: false,
      disableRegisterForm: false
    };
    this.loadAllModels();
    this.handleCancelRegister.bind(this);
    this.showRegisterForm.bind(this);
    this.submitRegisterForm.bind(this);
    this.setModelFileData.bind(this);
    this.handleNetworkError.bind(this);
  }

  public componentDidMount() {
    const { pagination } = this.state;
    this.fetch({ pagination });
  }

  public handleTableChange = (pagination, filters, sorter) => {
    this.fetch({
      sortField: sorter.field,
      sortOrder: sorter.order,
      pagination,
      ...filters,
    });
  };

  public fetch = (params = {}) => {
    this.setState({ loading: true });
    reqwest({
      url: config.modelURL,
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
        files: null
      });
    }).catch((error) =>{
      this.handleNetworkError();
    }

    );
  };

  public loadAllModels = () => {
    const targetUrl = config.modelURL;
    axios
      .get(targetUrl, {timeout:1000, timeoutErrorMessage: "The is not reachable"})
      .then((response) => {
        this.setState({ allModelInfo: response.data });
      })
      .catch((error) => {
        this.setState({ allModelInfo: [] });
      })
      .then(() => {
        // always executed
      });
  };

  public showRegisterForm = () => {
    this.setState({ showRegisterForm: true });
  };

  public handleCancelRegister = () => {
    this.setState({ showRegisterForm: false });
  };

  public submitRegisterForm = async (values) => {
    values.files = this.state.files;
    values.metric[values.metric.name] = values.metric.score;
    delete values.metric.name;
    delete values.metric.score;
    values.metric = JSON.stringify(values.metric);
    values.inputs = JSON.stringify(values.inputs);
    values.outputs = JSON.stringify(values.outputs);
    var formData = new FormData();
    for ( var key in values ) {
      formData.append(key, values[key]);
    }
    await axios.post(config.modelURL, formData);
    this.setState({ showRegisterForm: false });
    window.location.reload();    
  };

  public setModelFileData = (e) =>{
    this.setState({ files: e.target.files[0] });
  }

  public handleNetworkError() {
    this.setState({disableRegisterForm: true, loading: false})
    Modal.error({
      title: 'ModelHub Connection Error',
      content: <p>Please make sure your restful api <br /><a href={config.modelURL}>{config.modelURL}</a> <br />is avaliable</p>,
    });
  }

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
          <Button size="large" type="primary" onClick={this.showRegisterForm} disabled={this.state.disableRegisterForm}>
            Register Model
          </Button>
          <Modal
            title="Model Registration"
            visible={this.state.showRegisterForm}
            onOk={this.submitRegisterForm}
            onCancel={this.handleCancelRegister}
            width={1000}
            footer={null}
          >
            <Form.Item
              label="Model File"
              name="files"
              rules={[
                { required: true, message: 'Please select your model file!' },
              ]}
            >
              <Input type="file" onChange={this.setModelFileData}/>
            </Form.Item>
            <SchemaForm
              components={components}
              schema={registerSchema}
              style={{
                fontSize: 'medium',
              }}
              onSubmit={this.submitRegisterForm}
            >
              <FormButtonGroup>
                <Submit>Submit</Submit>
                <Reset>Reset</Reset>
              </FormButtonGroup>
            </SchemaForm>
          </Modal>

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
                  title="Converted Model Info"
                >
                  <Descriptions.Item
                    label="Model Name"
                  >
                    <Tag color="volcano">{record.architecture}</Tag>
                  </Descriptions.Item>
                  <Descriptions.Item
                    label="Converted Version"
                  >
                    <Tag color="blue">
                      {record.engine === 'PYTORCH' ? '' : record.framework}
                    </Tag>
                  </Descriptions.Item>
                  <Descriptions.Item
                    label="Serving Engine"
                  >
                    <Tag color="pink">
                      {(() => {
                        switch (record.engine) {
                          case 'TorchScript':
                            return 'PyTorch JIT + FastAPI';
                          case 'ONNX':
                            return 'ONNX Runtime + FastAPI';
                          case 'tensorrt':
                            return 'Triton Inference Server';
                          case 'TFS':
                            return 'TensorFlow Serving';
                          default:
                            return 'FastAPI';
                        }
                      })()}
                    </Tag>
                  </Descriptions.Item>
                  <Descriptions.Item
                    span={2}
                    label="Conversion State"
                  >
                    <a
                      style={{
                        fontSize: 20,
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
                  title="Profiling Results"
                >
                  <Descriptions.Item
                    label="Profiling Device"
                  >
                    <Tag color="geekblue">
                      {record.profile_result
                        ? record.profile_result.dynamic_results[0].device_name
                        : ''}
                    </Tag>
                  </Descriptions.Item>
                  <Descriptions.Item
                    label="Profiling Batch Size"
                  >
                    <Tag color="gold">
                      {record.profile_result
                        ? record.profile_result.dynamic_results[0].batch
                        : ''}
                    </Tag>
                  </Descriptions.Item>
                  <Descriptions.Item
                    label="Device Memory Utilization"
                  >
                    <Tag color="blue">
                      {record.profile_result
                        ? (
                            record.profile_result.dynamic_results[0].memory
                              .utilization * 100
                          ).toFixed(2)
                        : ''}{' '}
                      %
                    </Tag>
                  </Descriptions.Item>
                  <Descriptions.Item
                    label="Preprocess Latency"
                  >
                    <Tag color="cyan">
                      {record.profile_result
                        ? (
                            record.profile_result.dynamic_results[0].latency
                              .preprocess_latency.avg * 1000
                          ).toFixed(2)
                        : ''}{' '}
                      ms
                    </Tag>
                  </Descriptions.Item>
                  <Descriptions.Item
                    span={2}
                    label="All Batch Throughput"
                  >
                    <Tag color="geekblue">
                      {record.profile_result
                        ? record.profile_result.dynamic_results[0].throughput.inference_throughput.toFixed(
                            2
                          )
                        : ''}{' '}
                      req/sec
                    </Tag>
                  </Descriptions.Item>
                  <Descriptions.Item
                    label="P50 Latency"
                  >
                    <Tag color="gold">
                      {record.profile_result
                        ? (
                            record.profile_result.dynamic_results[0].latency
                              .inference_latency.p50 * 1000
                          ).toFixed(2)
                        : ''}{' '}
                      ms
                    </Tag>
                  </Descriptions.Item>
                  <Descriptions.Item
                    label="P95 Latency"
                  >
                    <Tag color="green">
                      {record.profile_result
                        ? (
                            record.profile_result.dynamic_results[0].latency
                              .inference_latency.p95 * 1000
                          ).toFixed(2)
                        : ''}{' '}
                      ms
                    </Tag>
                  </Descriptions.Item>
                  <Descriptions.Item
                    label="P99 Latency"
                  >
                    <Tag color="pink">
                      {record.profile_result
                        ? (
                            record.profile_result.dynamic_results[0].latency
                              .inference_latency.p99 * 1000
                          ).toFixed(2)
                        : ''}{' '}
                      ms
                    </Tag>
                  </Descriptions.Item>
                </Descriptions>
              </div>
            ),
            rowExpandable: () => true,
          }}
        />
      </Card>
    );
  }
}
