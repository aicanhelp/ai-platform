/**
 *  Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  openFuyao is licensed under Mulan PSL v2.
 *  You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  You may obtain a copy of Mulan PSL v2 at:
  
 *       http://license.coscl.org.cn/MulanPSL2
  
 *   THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *   EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *   MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *   See the Mulan PSL v2 for more details.
 */
import { useState, useEffect, useRef, useStore, useCallback } from 'openinula';
import { useTranslation } from 'react-i18next';
import { containerRouterPrefix } from '@/constant.js';
import '@/styles/pages/overview.less';
import { ResponseCode } from '@/common/constants';
import { Radio } from 'antd';
import { Link } from 'inula-router';
import { getIndexNodesListData, getIndexPodsListData } from '@/api/containerApi';
import nodeBlue1 from '@/assets/images/nodeBlue1.png';
import nodeBlue2 from '@/assets/images/nodeBlue2.png';
import nodeBlue3 from '@/assets/images/nodeBlue3.png';
import nodeGreen1 from '@/assets/images/nodeGreen1.png';
import nodeGreen2 from '@/assets/images/nodeGreen2.png';
import nodeGreen3 from '@/assets/images/nodeGreen3.png';
import universal4 from '@/assets/images/universal4.png';
import universal5 from '@/assets/images/universal5.png';

export default function NetworkLoad() {
  const { t } = useTranslation(['overview', 'common']);
  const themeStore = useStore('theme');
  const intervalStore = useStore('interval');
  const [isInflow, setIsInflow] = useState(true);
  const [selectedKey, setSelectedKeys] = useState('inflow');
  const [selectedNodeData, setSelectedNodeData] = useState([]);
  const [selectedPodData, setSelectedPodData] = useState([]);
  const timerRef = useRef(0);
  // Switch network load
  const handleSelectedNetKeys = useCallback((e) => {
    setSelectedKeys(e.target.value);
    setIsInflow(e.target.value === 'inflow');
  }, []);
  const nodeLoop = (instanceArr, _arr) => {
    const grouped = new Map();
    instanceArr.forEach(instance => {
      grouped.set(instance, []);
    });
    _arr.forEach(item => {
      const list = grouped.get(item.labels.instance);
      if (list) {
        list.push({ instance: item.labels.instance, value: item.sample.value });
      }
    });
    return instanceArr.map(instance => grouped.get(instance));
  };
  // Fetch network load node list
  const getWorkLoadNodesData = async (type) => {
    try {
      const res = await getIndexNodesListData(type);
      if (res.status === ResponseCode.OK) {
        let _arr = res.data.results[0].data.result;
        let instanceArr = [];
        _arr.forEach(item => {
          instanceArr.push(item.labels.instance);
        });
        instanceArr = [...new Set(instanceArr)];
        let itemArr = nodeLoop(instanceArr, _arr);
        let LastArr = [];
        for (let index = 0; index < itemArr.length; index++) {
          let sum = 0;
          for (let innnerIndex = 0; innnerIndex < itemArr[index].length; innnerIndex++) {
            sum += itemArr[index][innnerIndex].value;
          }
          LastArr.push({ ip: itemArr[index][0].instance, value: (sum / 1024).toFixed(2) });
        }
        let sortArr = LastArr.sort((a, b) => {
          return b.value - a.value;
        });
        setSelectedNodeData(sortArr.slice(0, 5));
      }
    } catch {
      setSelectedNodeData([]);
    }
  };

  // Fetch network load pod list
  const getWorkLoadPodsData = async (type) => {
    try {
      const res = await getIndexPodsListData(type);
      if (res.status === ResponseCode.OK) {
        let _arr = res.data.results[0].data.result;
        let instanceArr = [];
        _arr.forEach(item => {
          instanceArr.push({
            namespace: item.labels.namespace,
            pod: item.labels.pod,
            value: (item.sample.value / 1024).toFixed(2),
          });
        });
        let sortArr = instanceArr.sort((a, b) => {
          return b.value - a.value;
        });
        setSelectedPodData(sortArr.slice(0, 5));
      }
    } catch {
      setSelectedPodData([]);
    }
  };
  const timeSearch = () => {
    clearTimeout(timerRef.current);
    if (isInflow) {
      getWorkLoadNodesData('inflow');
      getWorkLoadPodsData('inflow');
    } else {
      getWorkLoadNodesData('flowOut');
      getWorkLoadPodsData('flowOut');
    }
    timerRef.current = setTimeout(() => {
      timeSearch();
    }, 30000);
  };
  useEffect(() => {
    clearTimeout(timerRef.current);
    if (intervalStore.$s.loginStatus) {
      timeSearch();
    }
    return () => clearTimeout(timerRef.current);
  }, [isInflow, intervalStore.$s.loginStatus]);

  return <div className="overview_flex overview_card_shadow" style={{ backgroundColor: themeStore.$s.theme === 'light' ? '#fff' : '#2a2d34ff' }}>
    <div className="overview_flex_net_box">
      <div className="overview_flex_net_title">
        <p className="overview_card_title" style={{ color: themeStore.$s.theme !== 'light' && '#f7f7f7' }}>{t('network.top5')}</p>
        <div className="overview_card_button">
          <Radio.Group
            value={selectedKey}
            onChange={handleSelectedNetKeys}
          >
            <Radio.Button value="inflow">{t('network.inflow')}</Radio.Button>
            <Radio.Button value="flowOut">{t('network.outflow')}</Radio.Button>
          </Radio.Group>
        </div>
      </div>

      <div className="overview_flex_net_title">
        <div className="overview_flex_net_left">
          <div className="overview_flex_net_left_node" style={{ color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7' }}>{t('network.node')}</div>
          <div className="overview_flex_net_left_nodeList">
            {selectedNodeData.map((item, index) => {
              return <div className="overview_flex_net_left_nodeListItem" key={item.ip}>
                <div className="overview_flex_net_left_nodeListItem_ip">
                  <div className="overview_flex_net_left_nodeListItem_ip_image">
                    {index === 0 && <img src={nodeBlue1} />}
                    {index === 1 && <img src={nodeBlue2} />}
                    {index === 2 && <img src={nodeBlue3} />}
                    {index === 3 && <img src={universal4} />}
                    {index === 4 && <img src={universal5} />}
                  </div>
                  <Link to={`/${containerRouterPrefix}/nodeManage/${item.ip}`}>{item.ip}</Link>
                </div>
                <div className="overview_flex_net_left_nodeListItem_speed">
                  <p className="overview_flex_net_left_nodeListItem_speed_value" style={{ color: themeStore.$s.theme !== 'light' && '#f7f7f7' }}>{item.value}</p>
                  <p className="overview_flex_net_left_nodeListItem_speed_unit" style={{ color: themeStore.$s.theme !== 'light' && '#9ea9b3' }}>MiBps</p>
                </div>
              </div>;
            })}

          </div>
        </div>
        <div className="overview_flex_net_right">
          <div className="overview_flex_net_right_pod" style={{ color: themeStore.$s.theme === 'light' ? '#333333' : '#f7f7f7' }}>{t('network.pod')}</div>
          <div className="overview_flex_net_right_podList">
            {selectedPodData.map((item, index) => {
              return <div className="overview_flex_net_right_podListItem" key={`${item.namespace}/${item.pod}`}>
                <div className="overview_flex_net_right_podListItem_ip">
                  <div className="overview_flex_net_right_podListItem_ip_image">
                    {index === 0 && <img src={nodeGreen1} />}
                    {index === 1 && <img src={nodeGreen2} />}
                    {index === 2 && <img src={nodeGreen3} />}
                    {index === 3 && <img src={universal4} />}
                    {index === 4 && <img src={universal5} />}
                  </div>
                  <Link to={`/${containerRouterPrefix}/workload/pod/${item.namespace}/${item.pod}`}>{item.pod.length > 40 ? item.pod.slice(0, 40).concat('...') : item.pod}</Link>
                </div>
                <div className="overview_flex_net_right_podListItem_speed">
                  <p className="overview_flex_net_right_nodeListItem_speed_value" style={{ color: themeStore.$s.theme !== 'light' && '#f7f7f7' }}>{item.value}</p>
                  <p className="overview_flex_net_right_nodeListItem_speed_unit" style={{ color: themeStore.$s.theme !== 'light' && '#9ea9b3' }}>MiBps</p>
                </div>
              </div>;
            })}
          </div>
        </div>
      </div>
    </div>
  </div>;
}