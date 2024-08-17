import { useState, useRef, useEffect } from "react";
import "./App.css";
import FeatureCard from "./FeatureCard";

type FeatureItem = {
	id: string;
	featureNumber: number;
};

export default function App() {
	const [features, setFeatures] = useState<FeatureItem[]>([
		{ id: crypto.randomUUID(), featureNumber: 5990 },
		{ id: crypto.randomUUID(), featureNumber: 10138 },
		{ id: crypto.randomUUID(), featureNumber: 1015 },
	]);
	const [newFeature, setNewFeature] = useState<string>("");
	const [isInputVisible, setIsInputVisible] = useState(false);
	const [isInputExpanding, setIsInputExpanding] = useState(false);
	const [isInputCollapsing, setIsInputCollapsing] = useState(false);
	const inputRef = useRef<HTMLInputElement>(null);

	const handleAddFeature = (e: React.KeyboardEvent<HTMLInputElement>) => {
		if (e.key === "Enter" && newFeature.trim() !== "") {
			const featureNumber = parseInt(newFeature);
			if (!isNaN(featureNumber)) {
				setFeatures([...features, { id: crypto.randomUUID(), featureNumber }]);
				setNewFeature("");
				collapseInput();
			}
		}
	};

	const removeFeature = (id: string) => {
		setFeatures(features.filter((feature) => feature.id !== id));
	};

	const expandInput = () => {
		setIsInputVisible(true);
		setIsInputExpanding(true);
		setTimeout(() => {
			setIsInputExpanding(false);
		}, 100); // Match this with the CSS transition duration
	};

	const collapseInput = () => {
		setIsInputCollapsing(true);
		setTimeout(() => {
			setIsInputVisible(false);
			setIsInputCollapsing(false);
			setNewFeature("");
		}, 100); // Match this with the CSS transition duration
	};

	useEffect(() => {
		if (isInputVisible && inputRef.current) {
			inputRef.current.focus();
		}
	}, [isInputVisible]);

	return (
		<div
			style={{ display: "flex", flexDirection: "column", alignItems: "center" }}
		>
			{features.map((feature) => (
				<FeatureCard
					key={feature.id}
					id={feature.id}
					featureNumber={feature.featureNumber}
					onDelete={removeFeature}
				/>
			))}
			<div className="add-feature-container">
				{!isInputVisible ? (
					<span className="add-icon" onClick={expandInput}>
						+
					</span>
				) : (
					<input
						ref={inputRef}
						type="text"
						value={newFeature}
						onChange={(e) => setNewFeature(e.target.value)}
						onKeyDown={handleAddFeature}
						onBlur={collapseInput}
						placeholder="Enter feature or search"
						className={`feature-input ${isInputExpanding ? "expanding" : ""} ${
							isInputCollapsing ? "collapsing" : ""
						}`}
					/>
				)}
			</div>
		</div>
	);
}
