import React from 'react';
import { Gitgraph, TemplateName, templateExtend } from '@gitgraph/react';
import axios from 'axios';
import { config, Link } from 'ice';
import { IGitData } from './utils/type';
import moment from 'moment';
import { Row, Col, List, Avatar, Button, Space } from 'antd';
import { EditOutlined } from '@ant-design/icons';
import './index.css'
const mockData = require('./utils/mock.json')
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

/**
 *  TODO parse multiple version tree
 * @param modelList list of model bo object
 */
const generateGitData = (modelList: []) => {
  let dataList: IGitData[] = []
  modelList.forEach((model: any) => {
    let data: IGitData = {
      author: {
        name: '',
        email: ''
      },
      hash: model.id.slice(7),
      refs: [],
      parents: model.parent_model_id == null ? ['root'] : [model.parent_model_id.slice(7)],
      subject: '',
      created_at: model.create_time
    }

    // original model
    if (model.parent_model_id == null) {
      data.subject = ' '
      data.refs.push('HEAD')
      data.refs.push('Main')
    } else if (['PYTORCH', 'None'].indexOf(model.engine) >= 0) {
      data.subject = ' '
      data.refs.push(`${model.name}/${model.dataset}`)
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
  public async componentDidMount() {
  }

  public async generateGitTree(gitgraph) {
    let res = await axios.get(config.modelURL);
    let modelList = res.data.reverse();
    this.setState({
      gitTreeData: generateGitData(modelList),
      modelData: modelList.filter(model => ['PYTORCH', 'None'].indexOf(model.engine) >= 0)
    })
    gitgraph.import(this.state.gitTreeData);
    console.log(this.state.gitTreeData)
  }

  public render() {
    return (
      <Row>
        <Col span={2}>
          <Gitgraph
            options={{ template: gitGraphOption }}
          >
            {async (gitgraph) => {
              // use real data
              await this.generateGitTree(gitgraph)
            }}
          </Gitgraph>
        </Col>
        <Col span={22}>
          <List
            size="large"
            itemLayout="horizontal"
            dataSource={this.state.modelData}
            renderItem={model => (
              <List.Item>
                <Space size="large">
                  <Button shape="round" size="large" style={{ width: 120 }}> {model.parent_model_id? model.dataset : 'Origin'} </Button>
                  <Button shape="round" size="large" style={{ width: 80 }}> v{model.version} </Button>
                  <Avatar size={48} src={`https://avatars.dicebear.com/4.5/api/identicon/:${model.creator}.svg`} />
                  {model.creator}
                  {moment(model.create_time).fromNow()}
                  <Link to={`/visualizer/${model.id}`}>
                  <EditOutlined style={{ fontSize: 30}}/>
                  </Link>                  
                </Space>

              </List.Item>
            )}
          />
        </Col>
      </Row>
    );
  };
}