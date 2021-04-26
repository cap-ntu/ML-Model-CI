import React from 'react';

const metadata = require('./onnx-metadata.json')

const nameList = metadata.map(x => x.name.toLocaleLowerCase());
const categories = metadata.map(x => x.category.toLocaleLowerCase());

type CustomNodeLabelProps = {
  label: string;
  shape: string;
  onClick: () => void;
}

export const CustomNodeLabel: React.FC<CustomNodeLabelProps> = ({ label, shape, onClick }) => {
  const index = nameList.indexOf(label.toLocaleLowerCase());
  let category = '';
  if(index !== -1){
    category = categories[index]
  }
  return (
    <div
      onClick={onClick} 
      style={{textAlign: 'center'}}>
      <div className={`node-item-type-${category}`} style={{ paddingLeft: 5, paddingRight: 5}}><p>{label}</p></div>
      {shape ? <div style={{marginTop: -10}}><p>&lt;{shape.join('x')}&gt;</p></div>: '' }
    </div>
  );
};

