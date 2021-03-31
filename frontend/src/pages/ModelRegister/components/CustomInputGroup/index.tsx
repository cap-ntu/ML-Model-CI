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
      <Select 
      defaultValue="acc" 
      style={{ width: '50%' }} 
      onChange={
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
      style={{ width: '50%' }} 
      defaultValue={0.76}
      min={0}
      max={1}
      step={0.01}
      onChange={
        (value)=>{
          props.mutators.pop()
          props.mutators.push({"score": value})
        }
      } />
    </Input.Group>
  )
}
CustomInputGroup.isFieldComponent = true
export default CustomInputGroup;
