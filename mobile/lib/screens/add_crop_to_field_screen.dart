import 'package:flutter/material.dart';
import '../models/crop_data.dart';

class AddCropToFieldScreen extends StatefulWidget {
  final String cropName;
  final String maturityDays;

  const AddCropToFieldScreen({
    super.key,
    required this.cropName,
    required this.maturityDays,
  });

  @override
  State<AddCropToFieldScreen> createState() => _AddCropToFieldScreenState();
}

class _AddCropToFieldScreenState extends State<AddCropToFieldScreen> {
  String? _selectedField;
  final List<String> _fields = ['Field 1', 'Field 2', 'Field 3'];

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
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              height: 300,
              width: double.infinity,
              decoration: BoxDecoration(
                color: Colors.grey[700],
                borderRadius: BorderRadius.circular(8),
              ),
              // Placeholder for map or image
            ),
            const SizedBox(height: 24),
            const Text(
              'Add Crop to a Field :',
              style: TextStyle(
                color: Colors.white,
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              decoration: BoxDecoration(
                color: const Color(0xFF3E3E3E),
                borderRadius: BorderRadius.circular(8),
              ),
              child: DropdownButtonHideUnderline(
                child: DropdownButton<String>(
                  value: _selectedField,
                  hint: const Text(
                    'Fields',
                    style: TextStyle(color: Colors.white70),
                  ),
                  isExpanded: true,
                  dropdownColor: const Color(0xFF3E3E3E),
                  style: const TextStyle(color: Colors.white),
                  icon: const Icon(
                    Icons.keyboard_arrow_down,
                    color: Colors.white,
                  ),
                  items: _fields.map((String field) {
                    return DropdownMenuItem<String>(
                      value: field,
                      child: Text(field),
                    );
                  }).toList(),
                  onChanged: (String? newValue) {
                    setState(() {
                      _selectedField = newValue;
                    });
                  },
                ),
              ),
            ),
            const Spacer(),
            Center(
              child: ElevatedButton(
                onPressed: _selectedField == null
                    ? null
                    : () {
                        final cropData = CropData(
                          name: widget.cropName,
                          statusColor: Colors.green,
                          progress: 0,
                          moisture: 'N/A',
                          temp: 'N/A',
                          sownDate: DateTime.now().toString().split(' ')[0],
                          lastIrrigation: 'N/A',
                          lastPesticide: 'N/A',
                          expectedYield: 'TBD', // Calculated from maturityDays
                        );
                        Navigator.pop(context, cropData);
                      },
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFFC5E1A5),
                  padding: const EdgeInsets.symmetric(
                    horizontal: 48,
                    vertical: 16,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30),
                  ),
                  disabledBackgroundColor: Colors.grey,
                ),
                child: const Text(
                  'Add Crop',
                  style: TextStyle(
                    color: Colors.black,
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
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
}
