import React from 'react';
import { Gitgraph, TemplateName, templateExtend } from '@gitgraph/react';
import axios from 'axios';
import { config } from 'ice';
import { IGitData } from './utils/type';
import moment from 'moment';

const mockData = require('./utils/mock.json')
const gitGraphOption = templateExtend(TemplateName.Metro, {
  commit: {
    message: {
      displayAuthor: true,
      displayHash: false
    },
  },
});

/**
 *  TODO parse multiple version tree
 * @param modelList list of model bo object
 */
const generateGitData = (modelList: []) => {
  let dataList: IGitData[] = []
  modelList.forEach( (model: any) => {
    let data: IGitData = {
      author: {
        name: model.creator,
        email: moment(model.create_time).fromNow()
      },
      hash: model.id.slice(7),
      refs: [],
      parents: model.parent_model_id == null? ['root'] : [model.parent_model_id.slice(7)],
      subject: '',
      created_at: model.create_time
    }
  
    // original model
    if (model.parent_model_id == null) {
      data.subject = `${model.name} Original`
      data.refs.push('HEAD')
      data.refs.push('Main')
    } else if (['PYTORCH', 'None'].indexOf(model.engine) >= 0){
      data.refs.push(`${model.name}/${model.dataset}`)
      data.subject = `Finetuned on ${model.dataset} dataset`
      if (model.version >= 4) {
        // Finetuned model
        data.subject = `[${model.name}][${model.dataset}] v${model.version}`
      }
    }
    // overlook model varient
    if(data.subject){
      dataList.push(data)
    }
  })

  return dataList
}


export default class VersionTree extends React.Component<{}, any> {
  constructor(props) {
    super(props);
    this.state = {
      gitTreeData: []
    };
  };
  public componentDidMount() {}

  public async generateGitTree(gitgraph) {
    const res = await axios.get(config.modelURL);
    // res.data.reverse().sort((a, b) => new Date(b.create_time) - new Date(a.create_time))
    this.setState({gitTreeData: generateGitData(res.data.reverse())})
    gitgraph.import(this.state.gitTreeData);
    console.log(this.state.gitTreeData)
  }

  public render() {
    return (
      <Gitgraph
        options={{ template: gitGraphOption }}
      >
        {async (gitgraph) => {
          // use real data
          // await this.generateGitTree(gitgraph)
          gitgraph.import(mockData);
        }}
      </Gitgraph>
    );
  };
}