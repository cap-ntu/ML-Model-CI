import React from 'react';
import { Input, Select, InputNumber } from 'antd';
const { Option } = Select;
function CustomInputGroup(props) {
  if(!props.mutators.exist(0)){
    props.mutators.push({"name":"acc"})
    props.mutators.push({"score": 0.76})
  }
  return (
    <Input.Group compact>
      <Select defaultValue="acc" onChange={
        (value)=>{
          props.mutators.shift()
          props.mutators.unshift({"name": value})
        }
        }>
        <Option value="acc">acc</Option>
        <Option value="mAp">mAp</Option>
        <Option value="IoU">IoU</Option>
      </Select>
      <InputNumber 
      defaultValue={76}
      min={0}
      max={100}
      formatter={value => `${value}%`}
      parser={value => value.replace('%', '')}
      onChange={
        (value)=>{
          props.mutators.pop()
          props.mutators.push({"score": value/100})
        }
      } />
    </Input.Group>
  )
}
CustomInputGroup.isFieldComponent = true
export default CustomInputGroup;
