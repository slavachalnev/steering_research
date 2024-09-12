// import React from "react";
import { BarChart, Bar, ResponsiveContainer } from "recharts";

const SimpleBarChart = ({ data }: { data: any }) => {
	return (
		<ResponsiveContainer width={450} height={200}>
			<BarChart data={data} margin={{ top: 5, right: 0, left: 0, bottom: 5 }}>
				<Bar dataKey="value" fill="#596CF8" />
			</BarChart>
		</ResponsiveContainer>
	);
};

const App = () => {
	const specificData = [
		1.3668, 5.7879e-1, 3.8011e-1, 2.5092e-2, 1.6172e-2, 4.2651e-3, 9.9414e-4,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, -1.2744e-3, -1.2905e-3, -3.6128e-3, -6.3863e-3,
		-2.0443e-2, -2.1909e-2, -2.3005e-2, -2.562e-2, -2.9994e-2, -3.49e-2,
		-3.9909e-2, -4.0954e-2, -4.2674e-2, -6.7345e-2, -7.9852e-2, -9.2134e-2,
		-1.204e-1, -1.6205e-1, -1.8758e-1, -2.1065e-1, -3.0074e-1, -7.1721e-1,
		-1.2202,
	];

	const chartData = specificData.map((value, index) => ({ value, index }));

	return (
		<div
			className="p-4"
			style={{
				backgroundColor: "white",
			}}
		>
			<SimpleBarChart data={chartData} />
		</div>
	);
};

export default App;
