import React from 'react';
import { Gitgraph, TemplateName, templateExtend } from "@gitgraph/react";
import axios from 'axios';
import { config } from 'ice';

const gitGraphOption = templateExtend(TemplateName.Metro, {
  commit: {
    message: {
      displayAuthor: false,
      displayHash: false
    },
  },
});


const generateGitData = (model) => {
  let data = {
    author: {
      name: "",
      email: "",
    },
    committer: {
      name: "",
      email: "",
    }
  }
  data.hash = model.id;
  data.parents = [model.parent_model_id];
  data.refs = [];
  data.created_at = model.timestamp
  // original model
  if (model.parent_model_id == "") {
    data.parents = ["root"]
    data.subject = "Original Model"
  }

  if (['PYTORCH', 'TRT'].indexOf(model.engine) >= 0) {
    data.refs.push(`tag: v${model.version}`)
    if (model.version > 1) {
      // Finetuned model
      data.subject = `Finetuned on ${model.dataset} dataset`
      data.refs.push("Main")
    }
  } else {
    // converted model
    data.subject = `Convert to ${model.engine}`
    data.refs = [`${model.engine}-${model.version}`];
  }

  return data
}


export default class GitTree extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      gitTreeData: []
    };
  };
  componentDidMount() {
  }

  generateTree(gitgraph) {
    axios.get(config.modelURL)
      .then(res => {
        let modelList: [] = res.data.sort((a, b) => {
          return a.create_time - b.create_time;
        });
        for (var model of modelList) {
          this.setState(prevState => ({
            gitTreeData: [...prevState.gitTreeData, generateGitData(model)]
          }))
        }
        gitgraph.import(this.state.gitTreeData)
      })
  };

  public render() {
    return (
      <div>
        <Gitgraph
          options={{ template: gitGraphOption }}
        >
          {(gitgraph) => {
            // this.generateTree(gitgraph)
            var master = gitgraph.branch('PyTorch Model')
            master.commit("Original Model").tag("v1.0");
            master.branch("ONNX").commit(
              {
                subject: "Add feature",
                body: "More details about the feature…",
                tag: "v1.2",
                onClick(commit) {
                  alert(`Commit ${commit.hash} selected`);
                },
              }
            )
            master.branch("TorchScript").commit("Convert to TorchScript version")
            master.commit('Finetune on CIFAR-10 dataset').tag("v2.0")
            master.branch("CIFAR10-ONNX").commit("Convert to ONNX version")
            master.branch("CIFAR10-TorchScript").commit("Convert to TorchScript version")

          }}
        </Gitgraph>
      </div>
    );
  };
}