import 'package:flutter/material.dart';
import 'package:easy_localization/easy_localization.dart';
// import 'package:hive_flutter/hive_flutter.dart';
import '../models/crop_data.dart';
import 'package:hive_flutter/hive_flutter.dart';

class RecommendedFertilizerScreen extends StatefulWidget {
  final Map<String, dynamic> result;
  final String cropName;
  // final String maturityDays;

  const RecommendedFertilizerScreen({
    super.key,
    required this.result,
    required this.cropName,
    // required this.maturityDays,
  });

  @override
  State<RecommendedFertilizerScreen> createState() =>
      _RecommendedFertilizerScreenState();
}

class _RecommendedFertilizerScreenState
    extends State<RecommendedFertilizerScreen> {
  List<Map<String, dynamic>> _fields = [];
  String? _selectedField;
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadFields();
  }

  Future<void> _loadFields() async {
    try {
      final box = await Hive.openBox('fields');
      final fields = box.values.toList();
      if (mounted) {
        setState(() {
          _fields = fields.map((e) => Map<String, dynamic>.from(e)).toList();
          _isLoading = false;
        });
      }
    } catch (e) {
      debugPrint("Error loading fields: $e");
      if (mounted) setState(() => _isLoading = false);
    }
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
        title: Text(
          'recommended_fertilizers'.tr(),
          style: const TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildFertilizerCard(
              'nitrogen_n'.tr(),
              widget.result['recommended_N'],
            ),
            const SizedBox(height: 16),
            _buildFertilizerCard(
              'phosphorus_p'.tr(),
              widget.result['recommended_P'],
            ),
            const SizedBox(height: 16),
            _buildFertilizerCard(
              'potassium_k'.tr(),
              widget.result['recommended_K'],
            ),
            const SizedBox(height: 40),
            Text(
              'add_crop_to_field'.tr(),
              style: const TextStyle(color: Colors.white, fontSize: 16),
            ),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              decoration: BoxDecoration(
                color: const Color(0xFF3E3E3E),
                borderRadius: BorderRadius.circular(8),
              ),
              child: _isLoading
                  ? const Center(
                      child: Padding(
                        padding: EdgeInsets.all(8),
                        child: CircularProgressIndicator(),
                      ),
                    )
                  : DropdownButtonHideUnderline(
                      child: DropdownButton<String>(
                        value: _selectedField,
                        hint: Text(
                          'select_field'.tr(),
                          style: const TextStyle(color: Colors.white70),
                        ),
                        isExpanded: true,
                        dropdownColor: const Color(0xFF3E3E3E),
                        icon: const Icon(
                          Icons.keyboard_arrow_down,
                          color: Colors.white,
                        ),
                        style: const TextStyle(color: Colors.white),
                        items: _fields.map((field) {
                          final name = field['name'] as String;
                          return DropdownMenuItem<String>(
                            value: name,
                            child: Text(name),
                          );
                        }).toList(),
                        onChanged: (value) {
                          setState(() {
                            _selectedField = value;
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
                        final selectedFieldData = _fields.firstWhere(
                          (element) => element['name'] == _selectedField,
                          orElse: () => {},
                        );

                        double? lat;
                        double? lon;

                        if (selectedFieldData.isNotEmpty &&
                            selectedFieldData['location'] != null) {
                          final loc = selectedFieldData['location'];
                          if (loc is Map) {
                            lat = loc['lat'];
                            lon = loc['lon'];
                          }
                        }

                        if (lat == null || lon == null) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(
                              content: Text(
                                'Selected field has no location data. Please select a field with location.',
                              ),
                            ),
                          );
                          return;
                        }

                        final cropData = CropData(
                          name: "${widget.cropName} ($_selectedField)",
                          statusColor: Colors.green,
                          progress: 0,
                          moisture: 'N/A',
                          temp: 'N/A',
                          sownDate: DateTime.now().toString().split(' ')[0],
                          lastIrrigation: 'N/A',
                          lastPesticide: 'N/A',
                          expectedYield: 'TBD',
                          latitude: lat,
                          longitude: lon,
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
                ),
                child: Text(
                  'add_crop'.tr(),
                  style: const TextStyle(
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

  Widget _buildFertilizerCard(String label, dynamic value) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
      decoration: BoxDecoration(
        color: const Color(0xFF1E1E1E),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFC5E1A5), width: 1),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: const TextStyle(color: Colors.white, fontSize: 16),
          ),
          RichText(
            text: TextSpan(
              children: [
                TextSpan(
                  text: value.toString(),
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const TextSpan(
                  text: ' kg/h',
                  style: TextStyle(color: Colors.white70, fontSize: 14),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
