import BasicLayout from '@/layouts/BasicLayout';
import Dashboard from '@/pages/Dashboard';
import About from '@/pages/About';
import Setting from '@/pages/Setting';
import User from '@/pages/User';
import Visualization from '@/pages/Visualization';

const routerConfig = [
  {
    path: '/',
    component: BasicLayout,
    children: [
      {
        path: '/',
        exact: true,
        component: Dashboard,
      },
      {
        path: '/about',
        component: About,
      },
      {
        path: '/user',
        component: User,
      },
      {
        path: '/setting',
        component: Setting,
      },
      {
        path: '/visualization/:id',
        component: Visualization,
      },
    ],
  },
];
export default routerConfig;
