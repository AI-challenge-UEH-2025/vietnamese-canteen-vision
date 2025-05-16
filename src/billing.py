import pandas as pd
import os

class BillingSystem:
    def __init__(self, menu_path=None):
        """
        Initialize the billing system with a CSV menu file
        
        Args:
            menu_path: Path to the menu CSV file with columns: item, price, calories
        """
        # Default menu path if none is provided
        if menu_path is None:
            # Try to find the menu in a few common locations
            possible_paths = [
                'data/menu_info.csv',
                '../data/menu_info.csv',
                'web_ui/static/data/menu_info.csv',
                os.path.join(os.path.dirname(__file__), '../data/menu_info.csv')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    menu_path = path
                    break
            
            # If no menu file is found, create a default one
            if menu_path is None:
                self.create_default_menu()
                menu_path = 'data/menu_info.csv'
        
        # Load the menu
        try:
            self.menu = pd.read_csv(menu_path)
        except Exception as e:
            print(f"Error loading menu from {menu_path}: {e}")
            self.create_default_menu()
            self.menu = pd.read_csv('data/menu_info.csv')

    def calculate_bill(self, food_items):
        """
        Calculate the bill for a list of food items
        
        Args:
            food_items: List of food items
            
        Returns:
            bill_details: List of dictionaries with item, price, and calories
            total_cost: Total cost of all items
            total_calories: Total calories of all items
        """
        total_cost = 0
        total_calories = 0
        bill_details = []

        for item in food_items:
            if item in self.menu['item'].values:
                price = self.menu[self.menu['item'] == item]['price'].iloc[0]
                calories = self.menu[self.menu['item'] == item]['calories'].iloc[0]
                total_cost += price
                total_calories += calories
                bill_details.append({'item': item, 'price': price, 'calories': calories})
            else:
                # If the item is not in the menu, add it with price and calories set to 0
                # You might want to handle this differently in a production system
                print(f"Warning: Item '{item}' not found in menu")
                bill_details.append({'item': item, 'price': 0, 'calories': 0})

        return bill_details, total_cost, total_calories
    
    def create_default_menu(self):
        """Create a default menu file if none exists"""
        print("Creating default menu file...")
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Default menu items based on Vietnamese canteen food
        default_menu = {
            'item': [
                'com', 'banh mi', 'bap cai luoc', 'bap cai xao', 'bo xao', 
                'ca chien', 'ca chua', 'ca kho', 'ca rot', 'canh bau', 
                'canh bi do', 'canh cai', 'canh chua', 'canh rong bien', 
                'chuoi', 'dau bap', 'dau hu', 'dau que', 'do chua', 
                'dua hau', 'dua leo', 'ga chien', 'ga kho', 'kho qua', 
                'kho tieu', 'kho trung', 'nuoc mam', 'nuoc tuong', 'oi', 
                'ot', 'rau', 'rau muong', 'rau ngo', 'suon mieng', 
                'suon xao', 'thanh long', 'thit chien', 'thit luoc', 
                'tom', 'trung chien', 'trung luoc'
            ],
            'price': [
                10000, 20000, 15000, 15000, 25000,
                25000, 5000, 25000, 5000, 12000,
                12000, 12000, 12000, 12000,
                5000, 10000, 12000, 10000, 5000,
                15000, 5000, 25000, 25000, 15000,
                20000, 15000, 5000, 5000, 10000,
                3000, 12000, 15000, 12000, 25000,
                25000, 10000, 20000, 20000,
                30000, 8000, 8000
            ],
            'calories': [
                150, 250, 70, 85, 230,
                200, 25, 180, 50, 60,
                80, 70, 60, 65,
                105, 90, 120, 70, 40,
                85, 15, 280, 250, 110,
                220, 120, 30, 35, 70,
                5, 50, 70, 55, 200,
                210, 60, 230, 180,
                160, 90, 70
            ]
        }
        
        # Create the DataFrame and save to CSV
        menu_df = pd.DataFrame(default_menu)
        menu_df.to_csv('data/menu_info.csv', index=False)
        print("Default menu created at data/menu_info.csv")
        
        # Also create a copy in the web_ui static directory if it exists
        os.makedirs('web_ui/static/data', exist_ok=True)
        try:
            menu_df.to_csv('web_ui/static/data/menu_info.csv', index=False)
        except:
            pass

if __name__ == "__main__":
    # Test the billing system
    billing = BillingSystem()
    food_items = ['com', 'ga chien', 'canh chua', 'rau muong']
    bill_details, total_cost, total_calories = billing.calculate_bill(food_items)
    
    print("\nDetected Food Items:")
    for detail in bill_details:
        print(f"Item: {detail['item']}, Price: {detail['price']} VND, Calories: {detail['calories']} kcal")
    print(f"\nTotal Cost: {total_cost} VND")
    print(f"Total Calories: {total_calories} kcal")