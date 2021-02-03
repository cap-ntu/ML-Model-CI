import React from 'react';
import { ReactD3GraphViz } from '@hikeman/react-graphviz';
import { Row, Col, Card, Popover} from 'antd';
import axios from 'axios';
import { config } from 'ice';
import { GraphvizOptions } from 'd3-graphviz';
import GenerateSchema from 'generate-schema';
import Form from '@rjsf/material-ui';
import {FinetuneConfig,ModelStructure,DEFAULT_FINETUNE_CONFIG} from './utils/type'

const defaultOptions: GraphvizOptions = {
  fit: true,
  height: 700,
  width: 1000,
  zoom: true,
  zoomScaleExtent: [0.5, 20],
  zoomTranslateExtent: [[-2000,-20000],[1000, 1000]]
};


type VisualizerProps = { match: any};
type VisualizerState = { 
  graphData: string; 
  isLoaded: boolean; 
  modelStructure: ModelStructure;
  finetuneConfig: FinetuneConfig;
  visible: boolean;
  currentLayerName: string;
  currentLayerInfo: object;
  layerSchema: object;
  configSchema: object;
};

export default class Visualizer extends React.Component<VisualizerProps, VisualizerState> {
  constructor(props) {
    super(props);
    this.state = {
      graphData: '',
      isLoaded: false,
      modelStructure: {layer: {}, connection: {}},
      finetuneConfig: DEFAULT_FINETUNE_CONFIG,
      visible: false,
      currentLayerName: '',
      currentLayerInfo: {},
      layerSchema: {},
      configSchema: {
        'title' : 'Finetune Settings',
        'type' : 'object',
        'properties' : {
            'dataset_name': {
        'type' : 'string',
        default: 'CIFAR10',
              enum: ['CIFAR10']
            }
        }
      }
    };
    this.showLayerInfo = this.showLayerInfo.bind(this);
	  this.layerSubmit = this.layerSubmit.bind(this);
	  this.configSubmit = this.configSubmit.bind(this);
  }

  public async componentDidMount() {
    const id: string = this.props.match.params.id;
    const graph = await axios.get(`${config.visualizerURL}/${id}`);
    const struct = await axios.get(`${config.structureURL}/${id}`)
    this.setState({
		  isLoaded: true,
		  graphData: graph.data.dot,
		  modelStructure: struct.data
    });
  }

  /**
   * display layer settings form after click on the graph
   * TODO add default value
   * 
   * @param title layer title
   * 
   */
  public showLayerInfo = (title: string)=>{
    // TODO validation check of model layer name
    if(title.includes('.weight') && !title.includes('downsample')){
	  const layerName: string = title.replace('.weight','');
	  let layersInfo: object = this.state.modelStructure.layer;
	  if(layerName in layersInfo){
		  let layerInfo = { ...layersInfo[layerName] }
		  this.setState({ currentLayerInfo: layersInfo[layerName] })
		  this.setState({ currentLayerName: layerName})
		  delete layerInfo.op_;
		  delete layerInfo.type_;
		  const schema = GenerateSchema.json('Layer Parameters',layerInfo)
		  delete schema.$schema
		  for(let property in schema.properties){
			schema.properties[property].default = layerInfo[property]
		  }
		  this.setState({layerSchema: schema, visible: true, currentLayerName: layerName})
	  }	  
    }
  }

    /**
   * update model layer information after submit
   */
  public layerSubmit = (layer: any)=>{
	  let modifyMark = {op_: 'M'}
	  let newConfig = { ...this.state.currentLayerInfo, ...layer.formData, ...modifyMark }
	  let newStructure = {...this.state.modelStructure}
	  newStructure.layer[this.state.currentLayerName] =  newConfig
	  this.setState({ modelStructure: newStructure	})
	  // close the form
	  this.setState({visible: false});
}


  /**
   * submit finetune job and modified model structures
   */
  configSubmit = async ()=>{
	  // submit model structure
	  let layers = this.state.modelStructure.layer
	  let updatedLayers = Object.keys(layers).reduce(function(r, e) {
		if (layers[e].op_ != 'E') r[e] = layers[e]
		return r;
	  }, {})
	 // TODO add connection update info 
	  let submittedStructure: ModelStructure = {"layer": updatedLayers, "connection": {}}
	  let res = await axios.patch(`${config.structureRefractorURL}/${this.props.match.params.id}`,submittedStructure)
    // submit training job
    let newConfig = {...this.state.finetuneConfig}
    newConfig.model = res.data.id;
    res = await axios.post(config.trainerURL, newConfig)
    console.log(res.data.id)
  }


  /**
   * update finetine job config
   * TODO add more config options
   * @param e event
   */
  public handleConfigChange = (config: any)=>{
	  let newConfig = {...this.state.finetuneConfig}
	  newConfig.data_module = { ...config.formData, ...newConfig.data_module};
	  this.setState({ finetuneConfig: newConfig })
  }

  public render() {
    return (
      <Row gutter={16}>
        <Col span={16}>
          <Card title="Model Structure" bordered={false}>
            <Popover
              content={
			  <div style={{width: 300}}>
				  <Form 
				  schema={this.state.layerSchema}  
				  onSubmit={this.layerSubmit}
				  />
			  </div>
			  }
              title="Modify Layer Paramaters"
			  visible={this.state.visible}
			  placement="rightTop"
            >
              <div id="graphviz">
			  <ReactD3GraphViz 
			  dot={this.state.isLoaded ? this.state.graphData : 'graph {}'} 
			  options={defaultOptions} 
			  onClick={this.showLayerInfo}
			  />
              </div>
            </Popover>
          </Card>
        </Col>
        <Col span={8}>
			<Form 
			schema={this.state.configSchema}  
			onSubmit={this.configSubmit}
			onChange={this.handleConfigChange}
			/>
        </Col>
      </Row>
    );

  };
};