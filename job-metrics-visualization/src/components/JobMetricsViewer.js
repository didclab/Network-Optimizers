import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import { Chart, registerables } from 'chart.js';
import axios from 'axios';
import { useParams } from 'react-router-dom';

Chart.register(...registerables);

const JobMetricsViewer = () => {
  const { ownerId } = useParams();
  const [jobIds, setJobIds] = useState([]);
  const [selectedJobId, setSelectedJobId] = useState('');
  const [metricsCache, setMetricsCache] = useState({});
  const [selectedMetrics, setSelectedMetrics] = useState(['Throughput', 'Actions', 'Loss']);

  useEffect(() => {
    const fetchAllMetrics = async () => {
      try {
        const response = await axios.get(`${process.env.REACT_APP_API_BASE_URL}/${ownerId}`);
        setJobIds(Object.keys(response.data));
        setMetricsCache(response.data);
      } catch (error) {
        console.error('Error fetching job metrics:', error);
      }
    };
    fetchAllMetrics();
  }, [ownerId]);

  const handleJobChange = (event) => {
    setSelectedJobId(event.target.value);
  };

  const handleMetricChange = (event) => {
    const value = Array.from(event.target.selectedOptions, option => option.value);
    setSelectedMetrics(value);
  };

  const getJobDisplayName = (jobId, index) => {
    if (index === 0) {
      return `Job #${index + 1} (Latest)`;
    }
    return `Job #${index + 1}`;
  };

  const getOrdinalSuffix = (number) => {
    const j = number % 10;
    const k = number % 100;
    if (j === 1 && k !== 11) return 'st';
    if (j === 2 && k !== 12) return 'nd';
    if (j === 3 && k !== 13) return 'rd';
    return 'th';
  };

  const metricOptions = [
    { value: 'Throughput', label: 'Throughput', color: 'rgba(75,192,192,1)', yAxisID: 'y-throughput' },
    { value: 'Actions', label: 'Actions (Parallelism/Concurrency)', color: ['rgba(153,102,255,1)', 'rgba(255,99,132,1)'], yAxisID: 'y-actions' },
    { value: 'Loss', label: 'Loss', color: 'rgba(255,205,86,1)', yAxisID: 'y-loss' }
  ];

  const chartData = {
    labels: metricsCache[selectedJobId]?.epoch_data.map((_, index) => index + 1) || [],
    datasets: [
      ...(selectedMetrics.includes('Throughput') ? [{
        label: 'Throughput',
        data: metricsCache[selectedJobId]?.epoch_data.map(metric => metric.reward) || [],
        borderColor: 'rgba(75,192,192,1)',
        fill: false,
        yAxisID: 'y-throughput',
      }] : []),
      ...(selectedMetrics.includes('Actions') ? [
        {
          label: 'Action - Parallelism',
          data: metricsCache[selectedJobId]?.epoch_data.map(metric => metric.action[0]) || [],
          borderColor: 'rgba(153,102,255,1)',
          fill: false,
          yAxisID: 'y-actions',
        },
        {
          label: 'Action - Concurrency',
          data: metricsCache[selectedJobId]?.epoch_data.map(metric => metric.action[1]) || [],
          borderColor: 'rgba(255,99,132,1)',
          fill: false,
          yAxisID: 'y-actions',
        }
      ] : []),
      ...(selectedMetrics.includes('Loss') ? [{
        label: 'Loss',
        data: metricsCache[selectedJobId]?.epoch_data.map(metric => metric.loss) || [],
        borderColor: 'rgba(255,205,86,1)',
        fill: false,
        yAxisID: 'y-loss',
      }] : []),
    ],
  };

  const options = {
    responsive: true,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    scales: {
      ...(selectedMetrics.includes('Throughput') && {
        'y-throughput': {
          type: 'linear',
          display: true,
          position: 'left',
          title: {
            display: true,
            text: 'Throughput'
          },
          min: 0,
          ticks: {
            callback: (value) => {
              if (value < 10) return value.toFixed(2);
              if (value < 100) return value.toFixed(1);
              return value.toFixed(0);
            }
          }
        }
      }),
      ...(selectedMetrics.includes('Actions') && {
        'y-actions': {
          type: 'linear',
          display: true,
          position: 'left',
          title: {
            display: true,
            text: 'Actions (Parallelism/Concurrency)'
          },
          min: 0,
          max: 50,
          ticks: {
            callback: (value) => value.toFixed(0)
          }
        }
      }),
      ...(selectedMetrics.includes('Loss') && {
        'y-loss': {
          type: 'linear',
          display: true,
          position: 'left',
          title: {
            display: true,
            text: 'Loss'
          },
          min: 0,
          max: 2,
          ticks: {
            callback: (value) => value.toFixed(3)
          }
        }
      })
    },
    plugins: {
      tooltip: {
        callbacks: {
          title: function(context) {
            const epochNumber = context[0].dataIndex + 1;
            return `${epochNumber}${getOrdinalSuffix(epochNumber)} epoch`;
          },
          label: function(context) {
            let label = context.dataset.label || '';
            let value = context.parsed.y;
            
            if (label === 'Loss') {
              return `${label}: ${value.toFixed(3)}`;
            } else if (label.startsWith('Throughput')) {
              return `${label}: ${value.toFixed(1)}`;
            } else {
              return `${label}: ${value.toFixed(0)}`;
            }
          }
        }
      }
    }
  };

  return (
    <div className="metrics-wrapper-box">
      <div className="metrics-header">
        <h2>Job Metrics</h2>
        <div className="select-container">
          <select 
            className="metrics-select"
            onChange={handleJobChange} 
            value={selectedJobId}
          >
            <option value="" disabled>Select a job</option>
            {jobIds.map((jobId, index) => (
              <option key={jobId} value={jobId}>
                {getJobDisplayName(jobId, index)}
              </option>
            ))}
          </select>
          <select
            className="metrics-select"
            multiple
            value={selectedMetrics}
            onChange={handleMetricChange}
          >
            {metricOptions.map(option => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      </div>
      <div className="metrics-inner-box">
        {selectedJobId && <Line data={chartData} options={options} />}
      </div>
    </div>
  );
};

/* Testing with dummy data */

//   const [jobIds, setJobIds] = useState(['job1', 'job2', 'job3']);
//   const [selectedJobId, setSelectedJobId] = useState('');
//   const [selectedMetrics, setSelectedMetrics] = useState(['Throughput', 'Actions', 'Loss']);
//   const [metrics] = useState({
//     job1: [
//       { reward: 10, action: [1, 2], loss: 0.1 },
//       { reward: 20, action: [2, 3], loss: 0.2 },
//     ],
//     job2: [
//       { reward: 15, action: [1, 4], loss: 0.2 },
//       { reward: 25, action: [3, 6], loss: 0.15 },
//     ],
//     job3: [
//       { reward: 30, action: [2, 9], loss: 0.6 },
//       { reward: 35, action: [3, 10], loss: 0.8 },
//     ],
//   });

//   const getJobDisplayName = (jobId, index) => {
//     if (index === 0) {
//       return `Job #${index + 1} (Latest)`;
//     }
//     return `Job #${index + 1}`;
//   };

//   const metricOptions = [
//     { value: 'Throughput', label: 'Throughput', color: 'rgba(75,192,192,1)', yAxisID: 'y-throughput' },
//     { value: 'Actions', label: 'Actions (Parallelism/Concurrency)', color: ['rgba(153,102,255,1)', 'rgba(255,99,132,1)'], yAxisID: 'y-actions' },
//     { value: 'Loss', label: 'Loss', color: 'rgba(255,205,86,1)', yAxisID: 'y-loss' }
//   ];

//   const chartData = {
//     labels: metrics[selectedJobId]?.map((_, index) => index + 1) || [],
//     datasets: [
//       ...(selectedMetrics.includes('Throughput') ? [{
//         label: 'Throughput',
//         data: metrics[selectedJobId]?.map(metric => metric.reward) || [],
//         borderColor: 'rgba(75,192,192,1)',
//         fill: false,
//         yAxisID: 'y-throughput',
//       }] : []),
//       ...(selectedMetrics.includes('Actions') ? [
//         {
//           label: 'Action - Parallelism',
//           data: metrics[selectedJobId]?.map(metric => metric.action[0]) || [],
//           borderColor: 'rgba(153,102,255,1)',
//           fill: false,
//           yAxisID: 'y-actions',
//         },
//         {
//           label: 'Action - Concurrency',
//           data: metrics[selectedJobId]?.map(metric => metric.action[1]) || [],
//           borderColor: 'rgba(255,99,132,1)',
//           fill: false,
//           yAxisID: 'y-actions',
//         }
//       ] : []),
//       ...(selectedMetrics.includes('Loss') ? [{
//         label: 'Loss',
//         data: metrics[selectedJobId]?.map(metric => metric.loss) || [],
//         borderColor: 'rgba(255,205,86,1)',
//         fill: false,
//         yAxisID: 'y-loss',
//       }] : []),
//     ],
//   };

//   const options = {
//     responsive: true,
//     interaction: {
//       mode: 'index',
//       intersect: false,
//     },
//     scales: {
//       ...(selectedMetrics.includes('Throughput') && {
//         'y-throughput': {
//           type: 'linear',
//           display: true,
//           position: 'left',
//           title: {
//             display: true,
//             text: 'Throughput'
//           },
//           min: 0,
//           ticks: {
//             callback: (value) => {
//               if (value < 10) return value.toFixed(2);
//               if (value < 100) return value.toFixed(1);
//               return value.toFixed(0);
//             }
//           }
//         }
//       }),
//       ...(selectedMetrics.includes('Actions') && {
//         'y-actions': {
//           type: 'linear',
//           display: true,
//           position: 'left',
//           title: {
//             display: true,
//             text: 'Actions (Parallelism/Concurrency)'
//           },
//           min: 0,
//           max: 50,
//           ticks: {
//             callback: (value) => value.toFixed(0)
//           }
//         }
//       }),
//       ...(selectedMetrics.includes('Loss') && {
//         'y-loss': {
//           type: 'linear',
//           display: true,
//           position: 'left',
//           title: {
//             display: true,
//             text: 'Loss'
//           },
//           min: 0,
//           max: 2,
//           ticks: {
//             callback: (value) => value.toFixed(3)
//           }
//         }
//       })
//     },
//     plugins: {
//       tooltip: {
//         callbacks: {
//           title: function(context) {
//             const epochNumber = context[0].dataIndex + 1;
//             return `${epochNumber}${getOrdinalSuffix(epochNumber)} epoch`;
//           },
//           label: function(context) {
//             let label = context.dataset.label || '';
//             let value = context.parsed.y;
            
//             if (label === 'Loss') {
//               return `${label}: ${value.toFixed(3)}`;
//             } else if (label.startsWith('Throughput')) {
//               return `${label}: ${value.toFixed(1)}`;
//             } else {
//               return `${label}: ${value.toFixed(0)}`;
//             }
//           }
//         }
//       }
//     }
//   };

//   const getOrdinalSuffix = (number) => {
//     const j = number % 10;
//     const k = number % 100;
//     if (j === 1 && k !== 11) return 'st';
//     if (j === 2 && k !== 12) return 'nd';
//     if (j === 3 && k !== 13) return 'rd';
//     return 'th';
//   };

//   const handleJobChange = (event) => {
//     setSelectedJobId(event.target.value);
//   };

//   const handleMetricChange = (event) => {
//     const value = Array.from(event.target.selectedOptions, option => option.value);
//     setSelectedMetrics(value);
//   };

//   return (
//     <div className="metrics-wrapper-box">
//       <div className="metrics-header">
//         <h2>Job Metrics</h2>
//         <div className="select-container">
//           <select 
//             className="metrics-select"
//             onChange={handleJobChange} 
//             value={selectedJobId}
//           >
//             <option value="" disabled>Select a job</option>
//             {jobIds.map((jobId, index) => (
//               <option key={jobId} value={jobId}>
//                 {getJobDisplayName(jobId, index)}
//               </option>
//             ))}
//           </select>
//           <select
//             className="metrics-select"
//             multiple
//             value={selectedMetrics}
//             onChange={handleMetricChange}
//           >
//             {metricOptions.map(option => (
//               <option key={option.value} value={option.value}>
//                 {option.label}
//               </option>
//             ))}
//           </select>
//         </div>
//       </div>
//       <div className="metrics-inner-box">
//         {selectedJobId && <Line data={chartData} options={options} />}
//       </div>
//     </div>
//   );
// };

export default JobMetricsViewer;