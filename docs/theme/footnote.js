// Decorate numeric-only table cells with .numeric so the stylesheet right-aligns
// them. Runs once on DOMContentLoaded; pure progressive enhancement.
(function () {
    "use strict";

    function decorate() {
        var cells = document.querySelectorAll(".content main table td");
        var numeric = /^[+\-]?(?:\d{1,3}(?:[, _]\d{3})*|\d+)(?:\.\d+)?(?:[eE][+\-]?\d+)?\s*(?:%|×|x)?$/;
        for (var i = 0; i < cells.length; i++) {
            var t = cells[i].textContent.trim();
            if (t.length > 0 && numeric.test(t)) {
                cells[i].classList.add("numeric");
            }
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", decorate);
    } else {
        decorate();
    }
})();
