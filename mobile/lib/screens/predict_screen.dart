import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import '../services/yield_service.dart';

class PredictScreen extends StatefulWidget {
  const PredictScreen({super.key});

  @override
  State<PredictScreen> createState() => _PredictScreenState();
}

class _PredictScreenState extends State<PredictScreen> {
  final _yieldService = YieldService();
  final FlutterTts flutterTts = FlutterTts();

  // Form values
  double _rainfall = 100.0;
  double _temperature = 25.0;
  double _nitrogen = 50.0;
  double _phosphorus = 50.0;
  double _potassium = 50.0;
  double _ph = 6.5;
  String _soilType = 'Alluvial';

  final List<String> _soilTypes = ['Red', 'Black', 'Alluvial', 'Clay', 'Sandy'];

  bool _isLoading = false;
  Map<String, dynamic>? _predictionResult;

  void _predict() async {
    setState(() {
      _isLoading = true;
      _predictionResult = null;
    });

    final formData = {
      'rainfall': _rainfall,
      'temperature': _temperature,
      'nitrogen': _nitrogen,
      'phosphorus': _phosphorus,
      'potassium': _potassium,
      'ph': _ph,
      'soil_type': _soilType,
    };

    final result = await _yieldService.getPrediction(formData);

    setState(() {
      _isLoading = false;
      _predictionResult = result;
    });

    if (result == null) {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('Prediction Failed')));
    }
  }

  Future<void> _speak(String text) async {
    await flutterTts.setLanguage("en-US");
    await flutterTts.speak(text);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Enter Crop Details',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),

            _buildSlider(
              'Rainfall (mm)',
              _rainfall,
              0,
              500,
              (val) => setState(() => _rainfall = val),
            ),
            _buildSlider(
              'Temperature (°C)',
              _temperature,
              0,
              50,
              (val) => setState(() => _temperature = val),
            ),
            _buildSlider(
              'Nitrogen',
              _nitrogen,
              0,
              200,
              (val) => setState(() => _nitrogen = val),
            ),
            _buildSlider(
              'Phosphorus',
              _phosphorus,
              0,
              200,
              (val) => setState(() => _phosphorus = val),
            ),
            _buildSlider(
              'Potassium',
              _potassium,
              0,
              200,
              (val) => setState(() => _potassium = val),
            ),
            _buildSlider(
              'pH Level',
              _ph,
              0,
              14,
              (val) => setState(() => _ph = val),
            ),

            const SizedBox(height: 16),
            DropdownButtonFormField<String>(
              initialValue: _soilType,
              decoration: const InputDecoration(labelText: 'Soil Type'),
              items: _soilTypes.map((String value) {
                return DropdownMenuItem<String>(
                  value: value,
                  child: Text(value),
                );
              }).toList(),
              onChanged: (newValue) {
                setState(() {
                  _soilType = newValue!;
                });
              },
            ),

            const SizedBox(height: 24),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: _isLoading ? null : _predict,
                child: _isLoading
                    ? const CircularProgressIndicator()
                    : const Text('Predict Yield'),
              ),
            ),

            if (_predictionResult != null) ...[
              const SizedBox(height: 24),
              if (_predictionResult!['is_offline'] == true)
                Container(
                  padding: const EdgeInsets.all(8),
                  margin: const EdgeInsets.only(bottom: 8),
                  decoration: BoxDecoration(
                    color: Colors.orange.shade100,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Row(
                    children: const [
                      Icon(Icons.wifi_off, color: Colors.orange),
                      SizedBox(width: 8),
                      Expanded(
                        child: Text('You are offline - showing cached data'),
                      ),
                    ],
                  ),
                ),
              Card(
                color: Colors.green.shade50,
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      Text(
                        'Predicted Yield: ${_predictionResult!['yield'] ?? 'N/A'}',
                        style: const TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                          color: Colors.green,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Advisory: ${_predictionResult!['advisory'] ?? 'No advisory available.'}',
                        style: const TextStyle(fontSize: 16),
                      ),
                      const SizedBox(height: 8),
                      IconButton(
                        icon: const Icon(Icons.volume_up),
                        onPressed: () {
                          _speak(
                            'Predicted Yield is ${_predictionResult!['yield']}. Advisory: ${_predictionResult!['advisory']}',
                          );
                        },
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildSlider(
    String label,
    double value,
    double min,
    double max,
    ValueChanged<double> onChanged,
  ) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('$label: ${value.toStringAsFixed(1)}'),
        Slider(
          value: value,
          min: min,
          max: max,
          divisions: (max - min).toInt(),
          label: value.toStringAsFixed(1),
          onChanged: onChanged,
        ),
      ],
    );
  }
}
