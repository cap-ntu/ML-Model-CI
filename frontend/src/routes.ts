import BasicLayout from '@/layouts/BasicLayout';
import Dashboard from '@/pages/Dashboard';
import About from '@/pages/About';
import Setting from '@/pages/Setting';
import User from '@/pages/User';
import Visualizer from '@/pages/Visualizer';
import GitTree from '@/pages/GitTree';

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
        path: '/visualizer/:id',
        component: Visualizer,
      },
      {
        path: '/gittree',
        component: GitTree,
      },
    ],
  },
];
export default routerConfig;
