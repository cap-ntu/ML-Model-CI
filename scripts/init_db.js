db = db.getSiblingDB('modelci');
db.createUser({user: "modelci", pwd: "modelci@2020", roles: ["readWrite"]});
