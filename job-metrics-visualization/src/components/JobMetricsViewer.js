import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import axios from 'axios';
import { useParams } from 'react-router-dom';

const JobMetricsViewer = () => {
  const { ownerId } = useParams();
  const [jobIds, setJobIds] = useState([]);
  const [selectedJobId, setSelectedJobId] = useState('');
  const [metrics, setMetrics] = useState([]);

  useEffect(() => {
    const fetchJobIds = async () => {
      try {
        const response = await axios.get(`http://localhost:8000/api/metrics/${ownerId}`);
        setJobIds(Object.keys(response.data));
      } catch (error) {
        console.error('Error fetching job IDs:', error);
      }
    };
    fetchJobIds();
  }, [ownerId]);

  useEffect(() => {
    if (selectedJobId) {
      const fetchMetrics = async () => {
        try {
          const response = await axios.get(`http://localhost:8000/api/metrics/${ownerId}/${selectedJobId}`);
          setMetrics(response.data.epoch_data);
        } catch (error) {
          console.error('Error fetching job metrics:', error);
        }
      };
      fetchMetrics();
    }
  }, [ownerId, selectedJobId]);

  const handleJobChange = (event) => {
    setSelectedJobId(event.target.value);
  };

  const chartData = {
    labels: metrics.map((_, index) => index + 1),
    datasets: [
      {
        label: 'Reward',
        data: metrics.map(metric => metric.reward),
        borderColor: 'rgba(75,192,192,1)',
        fill: false,
      },
      {
        label: 'Action',
        data: metrics.map(metric => metric.action),
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
      {selectedJobId && <Line data={chartData} />}
    </div>
  );
};

export default JobMetricsViewer;