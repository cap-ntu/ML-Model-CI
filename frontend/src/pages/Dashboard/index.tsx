import React from 'react';
import {
  Button,
  Table,
  Card,
  Divider,
  Input as search,
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
import axios from 'axios';
import { config, Link } from 'ice';
import reqwest from 'reqwest';
import '../index.css';
import ExpandedRow from './components/ExpandedRow'
// eslint-disable-next-line @typescript-eslint/no-var-requires

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
      disableRegisterForm: false
    };
    this.loadAllModels();
    this.handleNetworkError = this.handleNetworkError.bind(this);
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
          <Button size="large" type="primary" disabled={this.state.disableRegisterForm}>
          <Link to="/modelregister">
            Register Model
          </Link>
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
              <ExpandedRow record={record}/>
            ),
            rowExpandable: () => true,
          }}
        />
      </Card>
    );
  }
}
