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
	};

	const collapseInput = () => {
		if (newFeature.trim() === "") {
			setIsInputVisible(false);
			setNewFeature("");
		}
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
				<>
					<FeatureCard
						key={feature.id}
						id={feature.id}
						featureNumber={feature.featureNumber}
						onDelete={removeFeature}
					/>
					<div style={{ height: "10px", width: "100%" }} />
				</>
			))}
			<div className="add-feature-container">
				<span
					className={`add-icon ${isInputVisible ? "hidden" : ""}`}
					onClick={expandInput}
				>
					+
				</span>
				<input
					ref={inputRef}
					type="text"
					value={newFeature}
					onChange={(e) => setNewFeature(e.target.value)}
					onKeyDown={handleAddFeature}
					onBlur={collapseInput}
					placeholder="Enter feature or search"
					className={`feature-input ${isInputVisible ? "visible" : ""}`}
				/>
			</div>
		</div>
	);
}
