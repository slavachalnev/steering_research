import { useEffect } from "react";
import "./App.css";

const predictData = async (data) => {
	// const url = "https://api.replicate.com/v1/predictions";
	const url = "/api";
	const api_key = process.env.REPLICATE_API_TOKEN;

	try {
		// Sending the initial request to the /predict endpoint
		let response = await fetch(url, {
			method: "POST",
			headers: {
				Authorization: `Bearer ${api_key}`,
				"Content-Type": "application/json",
			},
			body: JSON.stringify(data),
		});

		if (!response.ok) {
			throw new Error(`Initial request failed: ${response.statusText}`);
		}

		let responseData = await response.json();

		// Extracting the prediction ID for polling
		let predictionId = responseData.id;
		let status = responseData.status;

		// Polling the API until the prediction is complete
		while (status !== "succeeded" && status !== "failed") {
			await new Promise((resolve) => setTimeout(resolve, 5000)); // Wait for 5 seconds before polling again

			let pollUrl = `${url}/${predictionId}`;
			response = await fetch(pollUrl, {
				method: "GET",
				headers: {
					"Content-Type": "application/json",
				},
			});

			if (!response.ok) {
				throw new Error(`Polling request failed: ${response.statusText}`);
			}

			responseData = await response.json();
			status = responseData.status;
			console.log(`Prediction status: ${status}`);
		}

		// Return the final response
		if (status === "succeeded") {
			return {
				status: "succeeded",
				output: responseData.output,
			};
		} else {
			throw new Error(`Prediction failed: ${responseData.error}`);
		}
	} catch (error) {
		console.error("Error:", error);
		return {
			status: "failed",
			error: error.message,
		};
	}
};

function App() {
	useEffect(() => {
		// Example usage:
		const data = {
			version:
				"81f1819b16d8ec24b0365d72fe6563365acca245123d7d37d327d1f94313165a",
			input: {
				prompt: "The best city",
				steering: "[[10138, 40]]",
				n_samples: 1,
				batch_size: 1,
				max_new_tokens: 1,
			},
		};

		predictData(data).then((result) => {
			console.log(result);
		});
	}, []);

	return (
		<div>
			<div></div>
			<div className=""></div>
		</div>
	);
}

export default App;
