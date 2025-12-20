import 'package:flutter/material.dart';
<<<<<<< HEAD

class SelectCropScreen extends StatelessWidget {
  const SelectCropScreen({super.key});

  final List<Map<String, String>> crops = const [
    {'name': 'Sunflower', 'icon': 'assets/images/icons/Frame 264.png'},
    {'name': 'Soybean', 'icon': 'assets/images/icons/Frame 265.png'},
    {'name': 'Mustard', 'icon': 'assets/images/icons/Frame 266.png'},
    {'name': 'Groundnut', 'icon': 'assets/images/icons/Frame 267.png'},
    {'name': 'Sesame', 'icon': 'assets/images/icons/Frame 268.png'},
    {'name': 'Castor', 'icon': 'assets/images/icons/Frame 264 (1).png'},
    {'name': 'Safflower', 'icon': 'assets/images/icons/Frame 267 (1).png'},
    {'name': 'Niger', 'icon': 'assets/images/icons/Frame 267 (2).png'},
  ];

=======
import 'package:easy_localization/easy_localization.dart';
import 'add_crop_details_screen.dart';

class SelectCropScreen extends StatefulWidget {
  const SelectCropScreen({super.key});

  @override
  State<SelectCropScreen> createState() => _SelectCropScreenState();
}

class _SelectCropScreenState extends State<SelectCropScreen> {
  final List<String> _selectedCrops = [];

  final List<Map<String, dynamic>> crops = const [
    {
      'name': 'Sunflower',
      'icon': 'assets/images/icons/Frame 264.png',
      'color': Color(0xFFEBC25C),
    },
    {
      'name': 'Mustard',
      'icon': 'assets/images/icons/Frame 265.png',
      'color': Color(0xFFEBC25C),
    },
    {
      'name': 'Soyabean',
      'icon': 'assets/images/icons/Frame 266.png',
      'color': Color(0xFF96B65D),
    },
    {
      'name': 'Safflower',
      'icon': 'assets/images/icons/Frame 267.png',
      'color': Color(0xFFEBC25C),
    },
    {
      'name': 'Sesame',
      'icon': 'assets/images/icons/Frame 268.png',
      'color': Color(0xFFC69C6D),
    },
    {
      'name': 'Niger',
      'icon': 'assets/images/icons/Frame 264 (1).png',
      'color': Color(0xFFE0E5C1),
    },
    {
      'name': 'Groundnut',
      'icon': 'assets/images/icons/Frame 267 (1).png',
      'color': Color(0xFFC69C6D),
    },
    {
      'name': 'Castor',
      'icon': 'assets/images/icons/Frame 267 (2).png',
      'color': Color(0xFF96B65D),
    },
  ];

  void _toggleCrop(String cropName) {
    setState(() {
      if (_selectedCrops.contains(cropName)) {
        _selectedCrops.remove(cropName);
      } else {
        _selectedCrops.add(cropName);
      }
    });
  }

>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
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
<<<<<<< HEAD
        title: const Text(
          'Select Crops',
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
=======
        title: Text(
          'select_crops'.tr(),
          style: const TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
          ),
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
<<<<<<< HEAD
            const Text(
              'Add the crops that you want to add.',
              style: TextStyle(color: Colors.white70, fontSize: 16),
=======
            if (_selectedCrops.isNotEmpty) ...[
              SizedBox(
                height: 100,
                child: ListView.builder(
                  scrollDirection: Axis.horizontal,
                  itemCount: _selectedCrops.length,
                  itemBuilder: (context, index) {
                    final cropName = _selectedCrops[index];
                    final cropData = crops.firstWhere(
                      (c) => c['name'] == cropName,
                    );
                    return Padding(
                      padding: const EdgeInsets.only(right: 16.0),
                      child: Column(
                        children: [
                          Stack(
                            children: [
                              Container(
                                padding: const EdgeInsets.all(0),
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  border: Border.all(
                                    color: cropData['color'],
                                    width: 0,
                                  ),
                                  color: const Color(0xFF1E1E1E),
                                ),
                                child: Image.asset(
                                  cropData['icon'],
                                  height: 70,
                                  width: 70,
                                  errorBuilder: (context, error, stackTrace) =>
                                      const Icon(
                                        Icons.grass,
                                        color: Colors.white,
                                        size: 70,
                                      ),
                                ),
                              ),
                              Positioned(
                                right: 0,
                                top: 0,
                                child: GestureDetector(
                                  onTap: () => _toggleCrop(cropName),
                                  child: Container(
                                    padding: const EdgeInsets.all(2),
                                    decoration: const BoxDecoration(
                                      color: Colors.red,
                                      shape: BoxShape.circle,
                                    ),
                                    child: const Icon(
                                      Icons.close,
                                      size: 12,
                                      color: Colors.white,
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 8),
                          Text(
                            cropName,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ),
                    );
                  },
                ),
              ),
            ],
            Text(
              'add_crops_instruction'.tr(),
              style: const TextStyle(color: Colors.white70, fontSize: 16),
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
            ),
            const SizedBox(height: 24),
            Expanded(
              child: GridView.builder(
<<<<<<< HEAD
=======
                padding: const EdgeInsets.only(bottom: 100),
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
                gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: 3,
                  crossAxisSpacing: 16,
                  mainAxisSpacing: 24,
                  childAspectRatio: 0.8,
                ),
                itemCount: crops.length,
                itemBuilder: (context, index) {
                  final crop = crops[index];
<<<<<<< HEAD
                  return GestureDetector(
                    onTap: () {
                      Navigator.pop(context, crop['name']);
                    },
                    child: Column(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            border: Border.all(
                              color: const Color(0xFFEBC25C),
                              width: 2,
                            ),
                            color: const Color(0xFF1E1E1E),
                          ),
                          child: Image.asset(
                            crop['icon']!,
                            height: 40,
                            width: 40,
                            errorBuilder: (context, error, stackTrace) =>
                                const Icon(
                                  Icons.grass,
                                  color: Colors.white,
                                  size: 40,
                                ),
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          crop['name']!,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 14,
                          ),
                          textAlign: TextAlign.center,
                        ),
                      ],
=======
                  final isSelected = _selectedCrops.contains(crop['name']);
                  return GestureDetector(
                    onTap: () => _toggleCrop(crop['name']),
                    child: Opacity(
                      opacity: isSelected ? 0.5 : 1.0,
                      child: Column(
                        children: [
                          Container(
                            padding: const EdgeInsets.all(0),
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              border: Border.all(
                                color: crop['color'],
                                width: 0,
                              ),
                              color: const Color(0xFF1E1E1E),
                            ),
                            child: Image.asset(
                              crop['icon']!,
                              height: 70,
                              width: 70,
                              errorBuilder: (context, error, stackTrace) =>
                                  const Icon(
                                    Icons.grass,
                                    color: Colors.white,
                                    size: 70,
                                  ),
                            ),
                          ),
                          const SizedBox(height: 8),
                          Text(
                            crop['name']!,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 14,
                            ),
                            textAlign: TextAlign.center,
                          ),
                        ],
                      ),
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
<<<<<<< HEAD
=======
      floatingActionButton: _selectedCrops.isNotEmpty
          ? Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16.0),
              child: SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: () async {
                    final result = await Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => AddCropDetailsScreen(
                          cropName: _selectedCrops.join(", "),
                        ),
                      ),
                    );

                    if (result != null && context.mounted) {
                      Navigator.pop(context, result);
                    }
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFFC5E1A5),
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(30),
                    ),
                  ),
                  child: Text(
                    'next'.tr(),
                    style: const TextStyle(
                      color: Colors.black,
                      fontWeight: FontWeight.bold,
                      fontSize: 16,
                    ),
                  ),
                ),
              ),
            )
          : null,
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
    );
  }
}
