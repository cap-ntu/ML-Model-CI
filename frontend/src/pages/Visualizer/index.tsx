import React from 'react';
import { ReactD3GraphViz } from '@hikeman/react-graphviz';
import { Row, Col, Card, Popover} from 'antd';
import axios from 'axios';
import { config } from 'ice';
import { GraphvizOptions } from 'd3-graphviz';
import GenerateSchema from 'generate-schema';
import Form from '@rjsf/material-ui';
import {ModelStructure} from './utils/type'

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
    // TODO display model layer without weight
    if(title.includes('.weight') || title.includes('.bias')){
	  const layerName: string = title.replace('.weight','').replace('.bias','');
	  const layersInfo: object = this.state.modelStructure.layer;
	  if(layerName in layersInfo){
		  const layerInfo = { ...layersInfo[layerName] }
		  this.setState({ currentLayerInfo: layersInfo[layerName] })
		  this.setState({ currentLayerName: layerName})
		  delete layerInfo.op_;
		  delete layerInfo.type_;
		  const schema = GenerateSchema.json('Layer Parameters',layerInfo)
		  delete schema.$schema
		  for(const property in schema.properties){
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
	  const modifyMark = {op_: 'M'}
	  const newConfig = { ...this.state.currentLayerInfo, ...layer.formData, ...modifyMark }
	  const newStructure = {...this.state.modelStructure}
	  newStructure.layer[this.state.currentLayerName] =  newConfig
	  this.setState({ modelStructure: newStructure	})
	  // close the form
	  this.setState({visible: false});
  }


  /**
   * submit finetune job and modified model structures
   */
  public configSubmit = async ()=>{
	  // submit model structure
	  const layers = this.state.modelStructure.layer
	  const updatedLayers = Object.keys(layers).reduce(function(r, e) {
      if (layers[e].op_ !== 'E') r[e] = layers[e]
      return r;
	  }, {})
	 // TODO add connection update info 
	  const submittedStructure: ModelStructure = {'layer': updatedLayers, 'connection': {}}
	  const res = await axios.patch(`${config.structureRefractorURL}/${this.props.match.params.id}`,submittedStructure)
    console.log(res.data.id)
    // TODO submit training job
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
          />
        </Col>
      </Row>
    );

  };
};