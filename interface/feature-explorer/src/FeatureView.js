import React, { useEffect, useState, useRef } from "react";
import { fetchData } from "./utils";

const normalizeValues = (values) => {
	const sum = values.reduce((acc, val) => acc + val, 0);
	return values.map((val) => val / sum);
};

const FeatureDetailsRow = ({
	values,
	indices,
	index,
	handleRows,
	barRef,
	setFeatureNumber,
}) => {
	const rowRef = useRef();
	const barRefs = useRef([]);
	const updatedIndices = indices.slice(1);
	const normalizedValues = normalizeValues(values.slice(1));
	const [svgPath, setSvgPath] = useState(null);
	const color = "rgba(0, 50, 200, 0.3)";
	const greyHover = "rgb(175, 175, 175)";
	const blueHover = "rgb(0, 50, 200, .6)";

	useEffect(() => {
		if (barRef && rowRef.current) {
			const barRect = barRef.getBoundingClientRect();
			const rowRect = rowRef.current.getBoundingClientRect();
			const svgRect = rowRef.current.parentElement.getBoundingClientRect();

			// Calculate relative positions
			const startLeft = barRect.left - svgRect.left;
			const startRight = barRect.right - svgRect.left;
			const startY = barRect.bottom - svgRect.top;
			const endLeft = rowRect.left - svgRect.left;
			const endRight = rowRect.right - svgRect.left;
			const endY = rowRect.top - svgRect.top + 2;

			// Calculate control points for the Bézier curves
			const midY = (startY + endY) / 2;

			const path = `
            M ${startLeft},${startY}
            C ${startLeft},${midY} ${endLeft},${midY} ${endLeft},${endY}
            L ${endRight},${endY}
            C ${endRight},${midY} ${startRight},${midY} ${startRight},${startY}
            Z
          `;

			setSvgPath(path);
		}
	}, [barRef, rowRef, values, indices]);

	useEffect(() => {
		// Set color on mount
		if (barRef) {
			barRef.style.backgroundColor = color;
		}

		// Remove color on dismount
		return () => {
			if (barRef) {
				barRef.style.backgroundColor = "";
			}
		};
	}, [barRef]);

	return (
		<div className="feature-details-row">
			{svgPath && (
				<svg
					style={{
						position: "absolute",
						top: 0,
						left: 0,
						width: "100%",
						height: "100%",
						pointerEvents: "none",
					}}
				>
					<path d={svgPath} fill={color} stroke="transparent" />
				</svg>
			)}
			<div ref={rowRef} className="feature-details">
				{normalizedValues.map((value, i) => {
					return (
						<div
							key={i}
							ref={(el) => (barRefs.current[i] = el)}
							style={{
								width: `${value * 100}%`,
							}}
							className="feature-bar"
							// onDoubleClick={(ev) => {
							// 	handleRows(updatedIndices[i], barRefs.current[i], index, i);
							// }}
							onClick={(ev) => {
								// setFeatureNumber(updatedIndices[i]);
								handleRows(updatedIndices[i], barRefs.current[i], index, i);
							}}
						>
							{updatedIndices[i]}
						</div>
					);
				})}
			</div>
		</div>
	);
};

const FeatureDetails = ({ data, setFeatureNumber }) => {
	const [rows, setRows] = useState([data]);
	const [barRefs, setBarRefs] = useState([null]);

	useEffect(() => {
		console.log("rows:");
		console.log(rows);
	}, [rows]);

	const handleRows = async (feature, ref, rowIndex) => {
		if (!rows) return;
		if (barRefs.includes(ref)) {
			const index = barRefs.indexOf(ref);
			setBarRefs([...barRefs.slice(0, index)]);
			setRows([...rows.slice(0, index)]);
		} else {
			const data = await fetchData(feature);
			setBarRefs([...barRefs.slice(0, rowIndex + 1), ref]);
			setRows([...rows.slice(0, rowIndex + 1), data]);
		}
	};

	if (!data) {
		return <div>Loading...</div>;
	}

	return (
		<div className="feature-rows">
			{rows.map((row, i) => (
				<FeatureDetailsRow
					values={row.values}
					indices={row.indices}
					handleRows={handleRows}
					barRef={barRefs[i]}
					index={i}
					setFeatureNumber={setFeatureNumber}
				/>
			))}
		</div>
	);
};

const FeatureView = ({ feature, setFeatureNumber }) => {
	const [data, setData] = useState(null);

	const getData = async (feature) => {
		console.log("Fetching data for feature:", feature);
		const fetchedData = await fetchData(feature);
		setData(fetchedData);
	};

	useEffect(() => {
		getData(feature);
	}, []);

	useEffect(() => {
		getData(feature);
	}, [feature]);

	useEffect(() => {
		if (data) {
			console.log(feature);
			console.log("Data updated:", data);
			console.log("Values:", data.values);
			console.log("Indices:", data.indices);
		}
	}, [data]);

	return (
		<div className="feature-view row">
			{data ? (
				<div className="column">
					<h3 className="feature-title">Feature {feature}</h3>
					<FeatureDetails data={data} setFeatureNumber={setFeatureNumber} />
				</div>
			) : (
				<div>Loading...</div>
			)}
		</div>
	);
};

export default FeatureView;
