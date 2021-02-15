import React from 'react';
import { Gitgraph, TemplateName, templateExtend } from '@gitgraph/react';
import axios from 'axios';
import { config } from 'ice';
import { IGitData } from './utils/type';

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
    const data: IGitData = {
      author: {
        name: model.creator,
        email: new Date(model.create_time).toISOString()
      },
      hash: model.id.slice(7),
      refs: [],
      parents: model.parent_model_id == null? ['root'] : [model.parent_model_id.slice(7)],
      subject: '',
      created_at: model.create_time
    }
  
    // original model
    if (model.parent_model_id == null) {
      data.subject = 'Original Model'
      data.refs.push('HEAD')
      data.refs.push('Main')
    } else if (['PYTORCH', 'None'].indexOf(model.engine) >= 0){
      data.refs.push(`tag: v${model.version}`)
      if (model.version > 1) {
        // Finetuned model
        data.subject = `Finetuned on ${model.dataset} dataset`
        data.refs.push('Main')
      }
    } else {
      // converted model
      data.subject = `Convert to ${model.engine}`
      data.refs.push(`${model.engine}-${model.version}`);
    }
    dataList.push(data)
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
    res.data.reverse().sort((a, b) => new Date(b.create_time) - new Date(a.create_time))
    this.setState({gitTreeData: generateGitData(res.data)})
    gitgraph.import(this.state.gitTreeData);
    console.log(this.state.gitTreeData)
  }

  public render() {
    return (
      <Gitgraph
        options={{ template: gitGraphOption }}
      >
        {(gitgraph) => {
          gitgraph.import(mockData);
        }}
      </Gitgraph>
    );
  };
}