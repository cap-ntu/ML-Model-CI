import React from 'react';
import { Gitgraph, TemplateName, templateExtend } from '@gitgraph/react';
import axios from 'axios';
import { config, Link } from 'ice';
import { IGitData } from './utils/type';
import moment from 'moment';
import { Row, Col, Avatar, Tag, Space, Table } from 'antd';
import { EditOutlined } from '@ant-design/icons';
import './index.css'
const gitGraphOption = templateExtend(TemplateName.Metro, {
  commit: {
    message: {
      displayAuthor: false,
      displayHash: false,
    },
  },
  branch: {
    label: {
      display: false,
    },
  },
});

const mockData = require('./utils/mock.json')
/**
 *  TODO parse multiple version tree, display different variant of models
 * @param modelList list of model bo object
 */
const generateGitData = (modelList: []) => {
  let dataList: IGitData[] = []
  modelList.forEach((model: any) => {
    let data: IGitData = {
      author: {
        name: null,
        email: null
      },
      hash: model.id.slice(7),
      refs: [],
      parents: model.parent_model_id == null ? ['root'] : [model.parent_model_id.slice(7)],
      subject: null,
      created_at: model.create_time
    }
    // original model
    if (['PYTORCH', 'None'].indexOf(model.engine) >= 0){
      if (model.parent_model_id == null) {
        data.subject = ' '
        data.refs.push('HEAD')
      } else {
        data.subject = ' '
        data.refs.push(`${model.name}/${model.dataset}`)
      }
    }
    // overlook model varient
    if (data.subject) {
      dataList.push(data)
    }
  })

  return dataList
}


export default class VersionTree extends React.Component<{}, any> {
  constructor(props) {
    super(props);
    this.state = {
      gitTreeData: [],
      modelData: []
    };
  };

  /**
   * TODO: display optimized formats(variant) of models
   * @param gitgraph GitgraphUserApi Object
   */
  public async generateGitTree(gitgraph) {
    let res = await axios.get(config.modelURL);
    let modelList = res.data.sort((a, b) => new Date(b.create_time) - new Date(a.create_time))
    this.setState({
      gitTreeData: generateGitData(modelList),
      modelData: modelList.filter(model => ['PYTORCH', 'None'].indexOf(model.engine) >= 0)
    })
    gitgraph.import(this.state.gitTreeData);
    // gitgraph.import(mockData)
  }

  const columns = [
    {
      title: 'Tags',
      dataIndex: 'name',
      key: 'tag',
      render: (name, record) =>
        <div>
          {record.parent_model_id ? '' : <Tag color='gold'>Original</Tag>}
          <Tag>{record.dataset}</Tag>
        </div>,
    },
    {
      title: 'version',
      dataIndex: 'version',
      key: 'version',
      render: version => <Tag color='green'>v{version}</Tag>
    },
    {
      title: 'Creator',
      dataIndex: 'creator',
      key: 'creator',
      render: creator =>
        <div>
          <Space size="large">
            <Avatar size={48} src={`https://avatars.dicebear.com/4.5/api/identicon/:${creator}.svg`} />
            <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', width: 100, display: 'inline-block' }}>
              {creator}
            </span>
          </Space>
        </div>
    },
    {
      title: 'Create time',
      key: 'create_time',
      dataIndex: 'create_time',
      render: create_time => moment(create_time).fromNow()
    },
    {
      title: 'Action',
      dataIndex: 'id',
      key: 'action',
      render: (id) => (
        <Link to={`/visualizer/${id}`}>
          <EditOutlined style={{ fontSize: 30 }} />
        </Link>
      ),
    },
  ];
  public render() {
    return (
      <Row>
        <Col span={2} offset={2}>
          <Gitgraph
            options={{ template: gitGraphOption }}
          >
            {async (gitgraph) => {
              // use real data
              await this.generateGitTree(gitgraph)
            }}
          </Gitgraph>
        </Col>
        <Col span={12}>
          <Table 
          columns={this.columns} 
          dataSource={this.state.modelData} 
          pagination={false}
          />
        </Col>
      </Row>
    );
  };
}