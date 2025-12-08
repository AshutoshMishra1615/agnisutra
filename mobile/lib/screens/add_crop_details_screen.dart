import 'package:flutter/material.dart';
import '../models/crop_data.dart';
import 'add_crop_to_field_screen.dart';

class AddCropDetailsScreen extends StatefulWidget {
  final String cropName;

  const AddCropDetailsScreen({super.key, required this.cropName});

  @override
  State<AddCropDetailsScreen> createState() => _AddCropDetailsScreenState();
}

class _AddCropDetailsScreenState extends State<AddCropDetailsScreen> {
  final _maturityController = TextEditingController();
  final _districtController = TextEditingController();
  final _pinController = TextEditingController();
  final _stateController = TextEditingController();
  final _nitrogenController = TextEditingController();
  final _phosphorusController = TextEditingController();
  final _potassiumController = TextEditingController();
  final _irrigationController = TextEditingController();

  @override
  void dispose() {
    _maturityController.dispose();
    _districtController.dispose();
    _pinController.dispose();
    _stateController.dispose();
    _nitrogenController.dispose();
    _phosphorusController.dispose();
    _potassiumController.dispose();
    _irrigationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1E1E1E),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text(
          'Add Crop',
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildSectionTitle('CROP DETAILS'),
            const SizedBox(height: 16),
            _buildLabel('Maturity Days'),
            _buildTextField(_maturityController),
            const SizedBox(height: 24),
            _buildSectionTitle('WEATHER AND CLIMATE'),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _buildLabel('District'),
                      _buildTextField(_districtController),
                    ],
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _buildLabel('PIN'),
                      _buildTextField(_pinController),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            _buildLabel('State'),
            _buildTextField(_stateController),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: () {
                // TODO: Implement get location
              },
              icon: const Icon(Icons.my_location, color: Colors.black),
              label: const Text(
                'Get Current Location',
                style: TextStyle(color: Colors.black),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFFC5E1A5),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(20),
                ),
              ),
            ),
            const SizedBox(height: 24),
            _buildSectionTitle('FERTILIZER'),
            const SizedBox(height: 16),
            _buildLabel('Nitrogen (kg/hector)'),
            _buildTextField(_nitrogenController),
            const SizedBox(height: 16),
            _buildLabel('Phosphorus (kg/hector)'),
            _buildTextField(_phosphorusController),
            const SizedBox(height: 16),
            _buildLabel('Potassium (kg/hector)'),
            _buildTextField(_potassiumController),
            const SizedBox(height: 24),
            _buildSectionTitle('IRRIGATIONS'),
            const SizedBox(height: 16),
            _buildLabel('Number of Irrigations'),
            _buildTextField(_irrigationController),
            const SizedBox(height: 32),
            Center(
              child: ElevatedButton.icon(
                onPressed: () async {
                  final result = await Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => AddCropToFieldScreen(
                        cropName: widget.cropName,
                        maturityDays: _maturityController.text,
                      ),
                    ),
                  );
                  if (result != null && mounted) {
                    Navigator.pop(context, result);
                  }
                },
                icon: const Icon(Icons.auto_awesome, color: Colors.black),
                label: const Text(
                  'Predict Yield',
                  style: TextStyle(
                    color: Colors.black,
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                ),
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFFC5E1A5),
                  padding: const EdgeInsets.symmetric(
                    horizontal: 32,
                    vertical: 16,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 32),
          ],
        ),
      ),
    );
  }

  Widget _buildSectionTitle(String title) {
    return Text(
      title,
      style: const TextStyle(
        color: Colors.white,
        fontSize: 16,
        fontWeight: FontWeight.bold,
      ),
    );
  }

  Widget _buildLabel(String label) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8.0),
      child: Text(
        label,
        style: const TextStyle(color: Colors.white70, fontSize: 14),
      ),
    );
  }

  Widget _buildTextField(TextEditingController controller) {
    return TextField(
      controller: controller,
      style: const TextStyle(color: Colors.white),
      decoration: InputDecoration(
        filled: true,
        fillColor: const Color(0xFF3E3E3E),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: BorderSide.none,
        ),
        contentPadding: const EdgeInsets.symmetric(
          horizontal: 16,
          vertical: 14,
        ),
      ),
    );
  }
}
