import React from 'react';
import { 
  Row,
  Col, 
  Divider, 
  Checkbox, 
  Card, 
  Space, 
  Input,
  PageHeader, 
  Tooltip,
  Typography
} from 'antd';
import { config} from 'ice';
import {
  Select,
  Switch,
  NumberPicker,
  FormMegaLayout,
  FormLayout,
  ArrayTable,
} from '@formily/antd-components';
import { SchemaForm, FormButtonGroup, Submit, Reset } from '@formily/antd';
import axios from 'axios';
import { QuestionCircleOutlined } from '@ant-design/icons'
import CustomUpload from './components/CustomUpload'
import CustomInputGroup from './components/CustomInputGroup'

const registerSchema = require('./utils/schema.json');

const CustomDivider = props => {
  return (
    <Divider orientation="left" style={{ marginTop: 10}} > {props.title}</Divider>
  )
}

const CustomCheckbox = props => {
  return (
    <Checkbox onChange={(e)=>{props.onChange(e.target.checked)}}>
      <Space size='small'>
      {props.title} 
      <Tooltip title={`What is ${props.title}?`}>
            <Typography.Link href={props.link}><QuestionCircleOutlined /></Typography.Link>
      </Tooltip>
      </Space>
    </Checkbox>
  )
}

const CustonHeader = props => {
  return(
    <PageHeader
      ghost={false}
      title={props.title}
      extra={
        <FormButtonGroup>
          <Submit>Submit</Submit>
          <Reset>Reset</Reset>
        </FormButtonGroup>
      }
    />
  )
}

const components = {
  CustonHeader,
  CustomInputGroup,
  CustomUpload,
  CustomDivider,
  CustomCheckbox,
  Checkbox,
  Input,
  Select,
  Switch,
  NumberPicker,
  FormMegaLayout,
  FormLayout,
  ArrayTable
};

export default class ModelRegister extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      showRegisterForm: false,
      disableRegisterForm: false,
    };
    this.submitRegisterForm = this.submitRegisterForm.bind(this);
  }

  public submitRegisterForm = async (values) => {
    let newEntry = {};
    newEntry[values.metric[0].name] = values.metric[1].score;
    values.metric = JSON.stringify(newEntry);
    values.inputs = JSON.stringify(values.inputs);
    values.outputs = JSON.stringify(values.outputs);
    var formData = new FormData();
    for (var key in values) {
      formData.append(key, values[key]);
    }
    await axios.post(config.modelURL, formData);
    this.setState({ showRegisterForm: false });
    window.location.href = "/";
  };

  render() {
    return (
      <Row>
        <Col span={12} offset={6}>
          <Card>
            <SchemaForm
              id="modelRegisterForm"
              components={components}
              schema={registerSchema}
              style={{
                fontSize: 'medium',
              }}
              onSubmit={this.submitRegisterForm}
              onChange={(values) => console.log(values)}
            >
            </SchemaForm>
          </Card>
        </Col>
      </Row>
    );
  }
}
