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

  const chartData = {
    labels: metricsCache[selectedJobId]?.epoch_data.map((_, index) => index + 1) || [],
    datasets: [
      {
        label: 'Reward',
        data: metricsCache[selectedJobId]?.epoch_data.map(metric => metric.reward) || [],
        borderColor: 'rgba(75,192,192,1)',
        fill: false,
      },
      {
        label: 'Action',
        data: metricsCache[selectedJobId]?.epoch_data.map(metric => metric.action) || [],
        borderColor: 'rgba(153,102,255,1)',
        fill: false,
      },
    ],
  };

  return (
    <div>
      <h2>Job Metrics for Owner ID: {ownerId}</h2>
      <select onChange={handleJobChange} value={selectedJobId}>
        <option value="" disabled>Select a job</option>
        {jobIds.map(jobId => (
          <option key={jobId} value={jobId}>{jobId}</option>
        ))}
      </select>
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', width: '100%' }}>
        <div style={{ width: '80%', maxWidth: '1000px', height: '600px' }}>
          {selectedJobId && <Line data={chartData} />}
        </div>
      </div>
    </div>
  );
};

/* Testing with dummy data */

//   const [jobIds, setJobIds] = useState(['job1', 'job2', 'job3']);
//   const [selectedJobId, setSelectedJobId] = useState('');
//   const [metrics] = useState({
//     job1: [
//       { reward: 10, action: 1 },
//       { reward: 20, action: 2 },
//     ],
//     job2: [
//       { reward: 15, action: 1 },
//       { reward: 25, action: 3 },
//     ],
//     job3: [
//       { reward: 30, action: 2 },
//       { reward: 35, action: 3 },
//     ],
//   });

//   const chartData = {
//     labels: metrics[selectedJobId]?.map((_, index) => index + 1) || [],
//     datasets: [
//       {
//         label: 'Reward',
//         data: metrics[selectedJobId]?.map(metric => metric.reward) || [],
//         borderColor: 'rgba(75,192,192,1)',
//         fill: false,
//       },
//       {
//         label: 'Action',
//         data: metrics[selectedJobId]?.map(metric => metric.action) || [],
//         borderColor: 'rgba(153,102,255,1)',
//         fill: false,
//       },
//     ],
//   };

//   const handleJobChange = (event) => {
//     setSelectedJobId(event.target.value);
//   };

//   return (
//     <div>
//       <h2>Job Metrics for Owner ID: {ownerId}</h2>
//       <select onChange={handleJobChange} value={selectedJobId}>
//         <option value="" disabled>Select a job</option>
//         {jobIds.map(jobId => (
//           <option key={jobId} value={jobId}>{jobId}</option>
//         ))}
//       </select>
//       <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', width: '100%' }}>
//         <div style={{ width: '80%', maxWidth: '1000px', height: '600px' }}>
//           {selectedJobId && <Line data={chartData} />}
//         </div>
//       </div>
//     </div>
//   );
// };

export default JobMetricsViewer;